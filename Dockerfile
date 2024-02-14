
FROM python:3.9-slim-bullseye

# Create app directory
WORKDIR /flask-app

# Install app dependencies
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Bundle app source
COPY . .

EXPOSE 8080
CMD ["gunicorn","--config", "gunicorn_config.py", "app:app"]

#docker build -t flask-app:1.0.0 .
#docker run -d -p 8080:8080 flask-app:1.0.0