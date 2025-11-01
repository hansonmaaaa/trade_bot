FROM --platform=linux/amd64 python:3.9.10 AS build

WORKDIR /Equity_Screen 
COPY . /Equity_Screen

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "prod_nasdaq_data_download.py"]