import joblib
import requests
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, date, timedelta
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
MODEL_KEY = "long_term_consumption_model.pkl"
SCALER_KEY = "long_term_scaler.pkl"


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
            try:
                model = joblib.load(tmp.name)
            except Exception as e:
                logger.error(f"Joblib load error details: {str(e)}")
                logger.error(f"Error type: {type(e)}")
                logger.error(f"Error args: {e.args}")
                # Try loading with pickle to see if it's a joblib-specific issue
                try:
                    import pickle

                    model = pickle.load(open(tmp.name, "rb"))
                    logger.info("Pickle load succeeded where joblib failed")
                    return model
                except Exception as pickle_e:
                    logger.error(f"Pickle also failed: {str(pickle_e)}")
                    raise e

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


def load_model_from_s3_presigned(bucket, key):
    """Download model from S3 using pre-signed URL"""
    logger.info(f"Loading {key} from bucket {bucket} using pre-signed URL")

    try:
        # Generate pre-signed URL
        s3_client = boto3.client("s3", region_name=S3_REGION)
        url = s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=300,  # URL valid for 5 minutes
        )

        logger.info(f"Generated pre-signed URL for {key}")

        # Download using requests
        response = requests.get(url)
        response.raise_for_status()

        # Save to temporary file and load
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name

        # Load outside the with block to prevent file handle issues
        model = joblib.load(tmp_path)

        # Clean up
        os.unlink(tmp_path)

        return model
    except Exception as e:
        logger.error(f"Pre-signed URL loading method failed: {str(e)}", exc_info=True)
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


def get_prediction():
    """Get weather forecast and make consumption prediction for next 7 days"""
    logger.info("Starting prediction process")

    # Load model and scaler from S3 with multiple fallback methods
    model = None
    scaler = None

    methods = [
        ("primary", load_model_from_s3),
        ("alternative", load_model_from_s3_alternative),
        ("presigned", load_model_from_s3_presigned),
        ("pickle", load_model_from_s3_pickle),
    ]

    # Try different loading methods
    for method_name, method_func in methods:
        if model is None:
            try:
                logger.info(f"Trying {method_name} method to load model")
                model = method_func(S3_BUCKET, MODEL_KEY)
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
        loc = Nominatim(user_agent="Agent")
        getLoc = loc.geocode("Beirut, Lebanon")
        lat = getLoc.latitude
        lon = getLoc.longitude
        logger.info(f"Location coordinates: {lat}, {lon}")
    except Exception as e:
        logger.error(f"Error getting location: {str(e)}")
        raise

    # Get forecast data
    try:
        today = date.today()
        end_date = today + timedelta(days=7)
        logger.info(f"Getting forecast from {today} to {end_date}")

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": [
                "temperature_2m_mean",
                "temperature_2m_min",
                "temperature_2m_max",
                "precipitation_sum",
                "wind_direction_10m_dominant",
                "wind_speed_10m_mean",
                "surface_pressure_mean",
            ],
            "start_date": today.isoformat(),
            "end_date": end_date.isoformat(),
            "timezone": "auto",
        }

        # Send request to Open-Meteo
        logger.info("Sending request to Open-Meteo API")
        response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
        response.raise_for_status()  # Raise exception for non-2xx responses
        response = response.json()
        logger.info("Received forecast data successfully")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching weather forecast: {str(e)}")
        raise
    except json.JSONDecodeError:
        logger.error("Error parsing weather API response")
        raise

    # Process data
    try:
        logger.info("Processing forecast data")
        df = pd.DataFrame(response["daily"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.drop(0, axis=0).reset_index(drop=True)
        logger.info(f"Processing {len(df)} days of forecast data")

        # Format data for prediction
        df["time"] = pd.to_datetime(df["time"])
        df["Month"] = df["time"].dt.month
        df["Day"] = df["time"].dt.day
        df["Day_of_week"] = df["time"].dt.dayofweek

        df.drop("time", axis=1, inplace=True)

        df.rename(
            columns={
                "temperature_2m_mean": "tavg",
                "temperature_2m_min": "tmin",
                "temperature_2m_max": "tmax",
                "precipitation_sum": "prcp",
                "wind_direction_10m_dominant": "wdir",
                "wind_speed_10m_mean": "wspd",
                "surface_pressure_mean": "pres",
            },
            inplace=True,
        )

        columns_to_scale = ["tavg", "tmin", "tmax", "prcp", "wdir", "wspd", "pres"]
        df[columns_to_scale] = scaler.transform(df[columns_to_scale])
        logger.info("Data preprocessing completed")
    except Exception as e:
        logger.error(f"Error processing forecast data: {str(e)}")
        raise

    # Make prediction
    try:
        logger.info("Making prediction with model")
        predictions = model.predict(df.to_numpy())
        logger.debug(f"Raw predictions: {predictions}")

        # Format predictions
        results = {}
        for i, prediction in enumerate(predictions):
            day = i + 1
            results[f"Day {day}"] = float(prediction)

        logger.info("Prediction completed successfully")
        return results
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise


def lambda_handler(event, context):
    """AWS Lambda entry point"""
    logger.info(
        f"Lambda function invoked with request ID: {context.aws_request_id if context else 'local-test'}"
    )

    try:
        start_time = datetime.now()
        predictions = get_prediction()

        execution_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Lambda execution completed in {execution_time}s")

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(predictions),
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
