import joblib
import requests
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from geopy.geocoders import Nominatim
import boto3
import os
import tempfile
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# S3 configuration
S3_BUCKET = "equilux-machine-learning"
S3_REGION = "eu-west-1"
MODEL_KEY = "production_model.pkl"
SCALER_KEY = "production_scaler.pkl"

# DynamoDB configuration
USERS_TABLE = "Equilux_Users_Prosumers"


def load_model_from_s3(bucket, key):
    """Download model from S3 and load it"""
    logger.info(f"Loading {key} from bucket {bucket}")
    start_time = datetime.now()

    try:
        # Check if file exists first
        s3_client = boto3.client("s3", region_name=S3_REGION)
        try:
            # Try to get object metadata to verify it exists
            s3_client.head_object(Bucket=bucket, Key=key)
            logger.info(f"Object {key} exists in bucket {bucket}")
        except Exception as head_error:
            logger.error(
                f"Object {key} not found in bucket {bucket}: {str(head_error)}"
            )
            raise Exception(
                f"S3 object not found: {bucket}/{key}. Error: {str(head_error)}"
            )

        # Download and load the file
        with tempfile.NamedTemporaryFile() as tmp:
            logger.info(f"Downloading {key} to temporary file")
            s3_client.download_file(bucket, key, tmp.name)
            logger.info(f"Loading model from temporary file")
            model = joblib.load(tmp.name)

        logger.info(
            f"Successfully loaded {key} in {(datetime.now() - start_time).total_seconds()}s"
        )
        return model
    except Exception as e:
        logger.error(f"Error loading {key} from S3: {str(e)}", exc_info=True)
        raise


def load_model_from_s3_alternative(bucket, key):
    """Alternative method to load model from S3"""
    logger.info(f"Loading {key} from bucket {bucket} (alternative method)")

    try:
        s3_client = boto3.client("s3", region_name=S3_REGION)
        response = s3_client.get_object(Bucket=bucket, Key=key)

        # Load directly from the response body
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(response["Body"].read())
            tmp.flush()
            model = joblib.load(tmp.name)

        return model
    except Exception as e:
        logger.error(f"Alternative S3 loading method failed: {str(e)}", exc_info=True)
        raise


def load_model_from_s3_pickle(bucket, key):
    """Try loading with pickle directly instead of joblib"""
    logger.info(f"Loading {key} from bucket {bucket} using direct pickle")

    try:
        # Download the file
        s3_client = boto3.client("s3", region_name=S3_REGION)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            s3_client.download_file(bucket, key, tmp.name)
            tmp_path = tmp.name

        # Try loading with pickle
        import pickle

        with open(tmp_path, "rb") as f:
            model = pickle.load(f)

        # Clean up
        os.unlink(tmp_path)

        return model
    except Exception as e:
        logger.error(f"Pickle loading method failed: {str(e)}", exc_info=True)
        raise


def get_production_capacity(user_id):
    """Fetch the total_production_capacity for the user from DynamoDB"""
    logger.info(f"Fetching production capacity for user: {user_id}")

    dynamodb = boto3.resource("dynamodb", region_name=S3_REGION)
    table = dynamodb.Table(USERS_TABLE)

    try:
        response = table.get_item(Key={"user_id": user_id})

        if "Item" in response:
            # Get the total_production_capacity with a default of 1.0 if not found
            capacity = float(response["Item"].get("total_production_capacity", 1.0))
            logger.info(f"Found production capacity for user {user_id}: {capacity}")
            return capacity
        else:
            # If user not found, return default capacity of 1.0
            logger.warning(
                f"User {user_id} not found in {USERS_TABLE}, using default capacity"
            )
            return 1.0
    except Exception as e:
        logger.error(f"Error fetching production capacity: {str(e)}", exc_info=True)
        return 1.0


def get_prediction():
    """Get weather forecast and make production prediction for next 3 hours"""
    logger.info("Starting prediction process")

    # Load model and scaler from S3 with multiple fallback methods
    model = None
    scaler = None

    methods = [
        ("primary", load_model_from_s3),
        ("alternative", load_model_from_s3_alternative),
        ("pickle", load_model_from_s3_pickle),
    ]

    # Try different loading methods
    for method_name, method_func in methods:
        if model is None or scaler is None:
            try:
                logger.info(f"Trying {method_name} method to load model and scaler")
                if model is None:
                    model = method_func(S3_BUCKET, MODEL_KEY)
                if scaler is None:
                    scaler = method_func(S3_BUCKET, SCALER_KEY)
                logger.info(f"Successfully loaded with {method_name} method")
                break
            except Exception as e:
                logger.warning(f"{method_name} method failed: {str(e)}")
                continue

    if model is None or scaler is None:
        error_msg = "All methods to load models failed"
        logger.error(error_msg)
        raise Exception(error_msg)

    # Get location data
    try:
        logger.info("Getting location data for Beirut, Lebanon")
        loc = Nominatim(user_agent="Lambda-Agent")
        getLoc = loc.geocode("Beirut, Lebanon")
        lat = getLoc.latitude
        lon = getLoc.longitude
        logger.info(f"Location coordinates: {lat}, {lon}")
    except Exception as e:
        logger.error(f"Error getting location: {str(e)}", exc_info=True)
        raise

    # Get forecast data
    try:
        logger.info("Requesting weather forecast from Open-Meteo API")
        # Parameters for the Open-Meteo API
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "wind_speed_10m",
                "sunshine_duration",
                "surface_pressure",
                "direct_radiation",
                "temperature_2m",
                "relative_humidity_2m",
            ],
            "forecast_hours": 3,  # Changed to 3 for 3-hour prediction
            "timezone": "auto",
        }

        # Send request to Open-Meteo
        response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
        response.raise_for_status()
        response = response.json()
        logger.info("Received forecast data successfully")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching weather forecast: {str(e)}", exc_info=True)
        raise
    except json.JSONDecodeError:
        logger.error("Error parsing weather API response")
        raise

    # Process data
    try:
        logger.info("Processing forecast data")
        df = pd.DataFrame(response["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.drop(0, axis=0).reset_index(drop=True)

        # Format data for prediction
        df["time"] = pd.to_datetime(df["time"])

        # Extract Month, Day, Hour from 'time'
        df["Month"] = df["time"].dt.month
        df["Day"] = df["time"].dt.day
        df["Hour"] = df["time"].dt.hour

        # Drop the 'time' column
        df.drop("time", axis=1, inplace=True)

        # Rename columns
        df.rename(
            columns={
                "wind_speed_10m": "WindSpeed",
                "sunshine_duration": "Sunshine",
                "surface_pressure": "AirPressure",
                "direct_radiation": "Radiation",
                "temperature_2m": "AirTemperature",
                "relative_humidity_2m": "RelativeAirHumidity",
            },
            inplace=True,
        )

        # Reorder columns to match training data
        desired_order = [
            "Day",
            "Month",
            "Hour",
            "WindSpeed",
            "Sunshine",
            "AirPressure",
            "Radiation",
            "AirTemperature",
            "RelativeAirHumidity",
        ]
        df = df[desired_order]

        # Scale the data
        columns_to_scale = [
            "WindSpeed",
            "Sunshine",
            "AirPressure",
            "Radiation",
            "AirTemperature",
            "RelativeAirHumidity",
        ]
        df[columns_to_scale] = scaler.transform(df[columns_to_scale])
        logger.info("Data preprocessing completed")
        logger.debug(f"Processed data shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error processing forecast data: {str(e)}", exc_info=True)
        raise

    # Make prediction
    try:
        logger.info("Making prediction with model")
        data_array = df.iloc[0:1].to_numpy()  # Get first row as numpy array
        predictions = model.predict(data_array)[0]  # Get first prediction result
        logger.debug(f"Raw predictions: {predictions}")

        # Ensure non-negative predictions
        predicted_value1 = max(0, float(predictions[0]))
        predicted_value2 = max(0, float(predictions[1]))
        predicted_value3 = max(0, float(predictions[2]))

        # Format predictions
        results = {
            "Hour 1": predicted_value1,
            "Hour 2": predicted_value2,
            "Hour 3": predicted_value3,
        }

        logger.info("Prediction completed successfully")
        return results
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}", exc_info=True)
        raise


def lambda_handler(event, context):
    """AWS Lambda entry point"""
    logger.info(
        f"Lambda function invoked with request ID: {context.aws_request_id if context else 'local-test'}"
    )

    try:
        start_time = datetime.now()

        # Extract user ID (sub) from Cognito authorizer
        request_context = event.get("requestContext", {})
        authorizer = request_context.get("authorizer", {})

        # Get user_id from different possible locations in the event
        user_id = None

        # For API Gateway REST API with Cognito authorizer
        if "claims" in authorizer:
            user_id = authorizer.get("claims", {}).get("sub")
        # For API Gateway HTTP API with JWT authorizer
        else:
            user_id = authorizer.get("jwt", {}).get("claims", {}).get("sub")

        logger.info(f"Extracted user_id: {user_id}")

        # Get the base prediction (for 3 hours)
        predictions = get_prediction()
        logger.info(f"Base predictions: {predictions}")

        # If we have a user_id, fetch their production capacity and multiply
        if user_id:
            production_capacity = get_production_capacity(user_id)
            scaled_predictions = {
                key: value * production_capacity for key, value in predictions.items()
            }
            logger.info(
                f"User {user_id}: Scaled predictions with capacity {production_capacity}"
            )
        else:
            # If no user_id found, just use the base prediction
            scaled_predictions = predictions
            logger.warning("No user_id found in request, using base predictions")

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Lambda execution completed in {execution_time}s")

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(
                {"predictions": scaled_predictions, "base_predictions": predictions}
            ),
        }
    except Exception as e:
        logger.error(f"Error in lambda_handler: {str(e)}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps({"error": str(e)}),
        }
