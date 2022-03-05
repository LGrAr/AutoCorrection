FROM python:3.6.9

WORKDIR /dockerized_app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "Application/app.py"]