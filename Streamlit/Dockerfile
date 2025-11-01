FROM --platform=linux/amd64 python:3.9.10 AS build

# 
EXPOSE 8080

WORKDIR /Streamlit 
COPY . /Streamlit

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8080", "--server.address=0.0.0.0"]
