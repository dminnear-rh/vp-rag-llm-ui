FROM registry.access.redhat.com/ubi9/python-312:9.5

USER root

WORKDIR /app

COPY app.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

RUN chown -R 1001:0 .

USER 1001

CMD ["python", "app.py"]
