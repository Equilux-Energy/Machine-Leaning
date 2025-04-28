import joblib
import requests
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
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
MODEL_KEY = "short_term_consumption_model.pkl"
SCALER_KEY = "short_term_scaler.pkl"

# DynamoDB configuration
DYNAMODB_TABLE = "Equilux_Energy_Sensor_Data"


def load_model_from_s3(bucket, key):
    """Download model from S3 and load it"""
    logger.info(f"Loading model {key} from bucket {bucket}")
    start_time = datetime.now()

    s3_client = boto3.client("s3", region_name=S3_REGION)
    with tempfile.NamedTemporaryFile() as tmp:
        s3_client.download_file(bucket, key, tmp.name)
        model = joblib.load(tmp.name)

    elapsed_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Model {key} loaded successfully in {elapsed_time:.2f}s")
    return model


def get_previous_consumption(user_id):
    """Get the three most recent consumption readings for the user from DynamoDB"""
    logger.info(f"Fetching recent consumption data for user {user_id}")

    dynamodb = boto3.resource("dynamodb", region_name=S3_REGION)
    table = dynamodb.Table(DYNAMODB_TABLE)

    # Query only by user_id and let DynamoDB sort by the timestamp sort key
    response = table.query(
        KeyConditionExpression="userid = :user_id",
        ExpressionAttributeValues={":user_id": user_id},
        ScanIndexForward=False,  # Sort in descending order (most recent first)
        Limit=3,  # We only need the 3 most recent entries
    )

    items = response.get("Items", [])
    logger.info(f"Found {len(items)} recent items for user {user_id}")

    # Extract consumption values from messageData JSON
    consumption_values = []
    for item in items:
        try:
            # Get messageData - add debug to see what we're working with
            message_data = item.get("messageData", "{}")
            logger.debug(f"messageData type: {type(message_data).__name__}")

            # Handle based on what type we got
            if isinstance(message_data, dict):
                # Already a dictionary
                message_dict = message_data
            else:
                # It's a string that needs parsing
                message_dict = json.loads(message_data)

            consumption = float(message_dict.get("consumption", 0))
            timestamp = item.get("timestamp", "unknown")
            consumption_values.append(consumption)
            logger.debug(
                f"Parsed consumption value: {consumption} from timestamp: {timestamp}"
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # Added TypeError to catch attribute errors
            logger.warning(f"Failed to parse message data: {e}")
            consumption_values.append(0)

    # If we don't have enough values, pad with zeros
    while len(consumption_values) < 3:
        consumption_values.append(0)

    logger.info(f"Final consumption values to use: {consumption_values[:3]}")
    return consumption_values[:3]  # Return only the first 3 values


def get_prediction(user_id):
    """Get prediction for next three hours consumption"""
    logger.info(f"Starting prediction process for user {user_id}")

    try:
        # Load model and scaler from S3
        model = load_model_from_s3(S3_BUCKET, MODEL_KEY)
        scaler = load_model_from_s3(S3_BUCKET, SCALER_KEY)

        # Get previous consumption data
        prev_consumption = get_previous_consumption(user_id)
        prev_1hour, prev_2hour, prev_3hour = prev_consumption

        # Get location data for weather forecast
        logger.info("Getting geolocation data for weather forecast")
        loc = Nominatim(user_agent="Lambda-Agent")
        getLoc = loc.geocode("Beirut, Lebanon")
        lat = getLoc.latitude
        lon = getLoc.longitude
        logger.info(f"Location coordinates: {lat}, {lon}")

        # Get weather forecast data
        logger.info("Fetching weather forecast data")
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": [
                "temperature_2m",  # temp
                "dew_point_2m",  # dwpt
                "relative_humidity_2m",  # rhum
                "precipitation",  # prcp
                "wind_direction_10m",  # wdir
                "wind_speed_10m",  # wspd
                "pressure_msl",  # pres
            ],
            "forecast_hours": 3,  # Changed to 3 to get 3-hour forecast
            "timezone": "auto",
        }

        # Send request to Open-Meteo
        response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
        response.raise_for_status()
        response = response.json()
        logger.info("Weather data received successfully")

        # Process forecast data
        logger.info("Processing forecast data")
        df = pd.DataFrame(response["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        df = df.drop(0, axis=0).reset_index(drop=True)

        # Format data for prediction
        df["time"] = pd.to_datetime(df["time"])
        df["Month"] = df["time"].dt.month
        df["Day"] = df["time"].dt.day
        df["Hour"] = df["time"].dt.hour
        df["Day_of_week"] = df["time"].dt.dayofweek

        # Add previous consumption data directly as columns
        df["previous1hr"] = prev_1hour
        df["previous2hr"] = prev_2hour
        df["previous3hr"] = prev_3hour

        # Drop the time column
        df.drop("time", axis=1, inplace=True)

        # Rename columns to match model's expected input
        df.rename(
            columns={
                "temperature_2m": "temp",
                "dew_point_2m": "dwpt",
                "relative_humidity_2m": "rhum",
                "precipitation": "prcp",
                "wind_direction_10m": "wdir",
                "wind_speed_10m": "wspd",
                "pressure_msl": "pres",
            },
            inplace=True,
        )

        # Ensure columns are in the correct order for the model
        feature_columns = [
            "temp",
            "dwpt",
            "rhum",
            "prcp",
            "wdir",
            "wspd",
            "pres",
            "Month",
            "Day",
            "Hour",
            "Day_of_week",
            "previous1hr",
            "previous2hr",
            "previous3hr",
        ]

        # Reorder columns to match model expectations
        df = df[feature_columns]

        # Scale the data
        columns_to_scale = [
            "temp",
            "dwpt",
            "rhum",
            "prcp",
            "wdir",
            "wspd",
            "pres",
            "previous1hr",
            "previous2hr",
            "previous3hr",
        ]

        df[columns_to_scale] = scaler.transform(df[columns_to_scale])
        logger.info("Data preprocessing completed")
        logger.debug(f"Scaled data: {df}")

        # Make prediction
        logger.info("Making prediction with model")
        data_array = df.to_numpy()
        predictions = model.predict(data_array)

        # New model returns predictions for 3 hours
        logger.debug(f"Raw predictions: {predictions}")

        # Format predictions
        results = {}
        for i, prediction in enumerate(
            predictions[0]
        ):  # Using [0] as we're getting the first (and only) row of predictions
            hour = i + 1
            results[f"Hour {hour}"] = float(prediction)

        logger.info("Prediction completed successfully")
        return results

    except Exception as e:
        logger.error(f"Error in prediction process: {str(e)}", exc_info=True)
        raise


def lambda_handler(event, context):
    request_id = context.aws_request_id if context else "local-test"
    logger.info(f"Lambda function invoked with request ID: {request_id}")

    try:
        # Extract sub from the Cognito authorizer context
        request_context = event.get("requestContext", {})
        authorizer = request_context.get("authorizer", {})

        # Debug: Log what we received
        logger.debug(f"Request context: {json.dumps(request_context)}")
        logger.debug(f"Authorizer: {json.dumps(authorizer)}")

        # For API Gateway REST API
        if "claims" in authorizer:
            user_id = authorizer.get("claims", {}).get("sub")
            logger.info(f"Found sub in REST API authorizer: {user_id}")
        # For API Gateway HTTP API
        else:
            user_id = authorizer.get("jwt", {}).get("claims", {}).get("sub")
            logger.info(f"Found sub in HTTP API authorizer: {user_id}")

        if not user_id:
            logger.warning("User ID (sub) not found in authorization context")
            return {
                "statusCode": 400,
                "headers": {
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*",
                },
                "body": json.dumps(
                    {"error": "User ID (sub) not found in authorization context"}
                ),
            }

        # Get prediction - still use the same function, passing sub as the identifier
        start_time = datetime.now()
        prediction_value = get_prediction(user_id)
        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info(f"Request completed successfully in {elapsed:.2f}s")

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
            "body": json.dumps(prediction_value),
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
