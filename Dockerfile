FROM python:3.12.9
ENV PYTHONUNBUFFERED True

# Define build arguments
ARG AUTH_JWT_KEY
ARG HF_API_KEY

# Set environment variables from build arguments
ENV AUTH_JWT_KEY=$AUTH_JWT_KEY
ENV HF_API_KEY=$HF_API_KEY

RUN pip install --upgrade pip
COPY pyproject.toml .
COPY setup.py .
RUN pip install --no-cache-dir -e .

ENV APP_ROOT /root
WORKDIR $APP_ROOT

# Create .env file with environment variables
RUN echo "AUTH_JWT_KEY=$AUTH_JWT_KEY" > $APP_ROOT/.env && \
    echo "HF_API_KEY=$HF_API_KEY" >> $APP_ROOT/.env

COPY . $APP_ROOT

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]