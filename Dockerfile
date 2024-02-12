
FROM python:3.9-slim-bullseye

# Create app directory
WORKDIR /python-docker

# Install app dependencies
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Bundle app source
COPY . .

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]

#docker build -t flask-app .
#docker run -d -p 5000:5000 flask-app