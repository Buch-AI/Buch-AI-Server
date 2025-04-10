FROM python:3.12.9
ENV PYTHONUNBUFFERED True

# Install system dependencies including ImageMagick for MoviePy
RUN apt-get update && \
    apt-get install -y \
    imagemagick \
    libmagickwand-dev \
    ffmpeg \
    fonts-freefont-ttf \
    fontconfig && \
    rm -rf /var/lib/apt/lists/*

# Update font cache
RUN fc-cache -f

# Configure ImageMagick path
ENV MAGICK_HOME=/usr

# Configure ImageMagick policy to allow text operations and set resource limits
RUN sed -i 's/rights="none" pattern="@\*"/rights="read|write" pattern="@\*"/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="resource" name="width" value="16KP"/<policy domain="resource" name="width" value="64KP"/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="resource" name="height" value="16KP"/<policy domain="resource" name="height" value="64KP"/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="resource" name="memory" value="256MiB"/<policy domain="resource" name="memory" value="1GiB"/' /etc/ImageMagick-6/policy.xml && \
    sed -i 's/<policy domain="resource" name="disk" value="1GiB"/<policy domain="resource" name="disk" value="4GiB"/' /etc/ImageMagick-6/policy.xml

# Set proper permissions for ImageMagick and temporary directories
RUN chmod 777 /tmp && \
    chmod 644 /etc/ImageMagick-6/policy.xml && \
    mkdir -p /tmp/magick && \
    chmod 777 /tmp/magick

# Set working directory
ENV APP_ROOT /root
WORKDIR $APP_ROOT

# Define build arguments
ARG AUTH_JWT_KEY
ARG HF_API_KEY

# Set environment variables from build arguments
ENV AUTH_JWT_KEY=$AUTH_JWT_KEY
ENV HF_API_KEY=$HF_API_KEY

# Create .env file with environment variables
RUN echo "AUTH_JWT_KEY=$AUTH_JWT_KEY" > $APP_ROOT/.env && \
    echo "HF_API_KEY=$HF_API_KEY" >> $APP_ROOT/.env

# Copy only the requirements file first to leverage Docker cache
RUN pip install --upgrade pip
COPY pyproject.toml .
COPY setup.py .
RUN pip install --no-cache-dir -e .

# Copy only necessary files
COPY config.py $APP_ROOT/
COPY app/ $APP_ROOT/app/

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]