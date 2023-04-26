# Use the official Python image as the base image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /foliate

# Copy the application code to the container
COPY . .

# Install the app's dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Expose the port on which the app will listen
EXPOSE 5000
RUN ls

# Start gunicorn to serve the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "application:application"]
