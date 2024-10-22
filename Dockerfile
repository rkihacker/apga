FROM python:3.11
RUN apt update 
RUN apt install -y git
RUN git clone https://github.com/renqabs/apga.git /app
WORKDIR app
RUN pip install -r requirements.txt
EXPOSE 7860
CMD ["python", "main.py"]
