FROM python:3.10-slim

WORKDIR /app

RUN python3 -m venv .venv
RUN . .venv/bin/activate

COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 5000

ENV NAME bidboostrag

CMD ["python", "app.py"]