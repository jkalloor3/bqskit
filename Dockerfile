# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.9

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

# Install pip requirements
RUN apt-get update && apt-get install -y git libgfortran5  && python -m pip install --upgrade pip && python -m pip install -r requirements.txt && pip install -e .

