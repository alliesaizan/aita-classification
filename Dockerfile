FROM python:3.8-slim-buster
WORKDIR /app

COPY static static
COPY templates templates
COPY results results
COPY app.py app.py
COPY requirements_docker.txt requirements.txt

RUN pip3 install -r requirements.txt
EXPOSE 5000 
CMD ["python3","app.py"]