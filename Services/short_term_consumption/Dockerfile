FROM public.ecr.aws/lambda/python:3.12


# Install scikit-learn with a version that matches the model creation environment
# Important: Use the same version that was used to train and save your model
RUN pip install --no-cache-dir scikit-learn

# Install other dependencies
RUN pip install --no-cache-dir pandas joblib requests geopy boto3

# Copy function code
COPY lambda_function.py .

# Set the handler
CMD ["lambda_function.lambda_handler"]