FROM python:3.12.9
ENV PYTHONUNBUFFERED True

RUN pip install --upgrade pip
COPY pyproject.toml .
COPY setup.py .
RUN pip install --no-cache-dir -e .

ENV APP_ROOT /root
WORKDIR $APP_ROOT
COPY . $APP_ROOT

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]