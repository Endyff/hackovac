FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hackovac.py .
COPY examples examples/
CMD [ "streamlit", "run", "hackovac.py", "--server.address=0.0.0.0", "--server.port=4000", "--browser.serverAddress=127.0.0.1"  ]

EXPOSE 4000
