FROM python:3.12.9
ENV PYTHONUNBUFFERED True

RUN pip install --upgrade pip
COPY pyproject.toml .
COPY setup.py .
RUN pip install --no-cache-dir -e .

ENV APP_ROOT /root
WORKDIR $APP_ROOT
COPY . $APP_ROOT

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]