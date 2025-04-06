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

# Set environment variables
ENV PYTHONPATH=/app
ENV MAGICK_HOME=/usr
ENV ENV=p

# Set ImageMagick policy to allow PDF operations
COPY policy.xml /etc/ImageMagick-6/policy.xml

# Set proper permissions for ImageMagick and temporary directories
RUN chmod 777 /tmp && \
    chmod 644 /etc/ImageMagick-6/policy.xml && \
    mkdir -p /tmp/magick && \
    chmod 777 /tmp/magick

# Set working directory
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY pyproject.toml .
COPY setup.py .
RUN pip install --no-cache-dir -e .

# Copy the entire project
COPY . .

# Run the video generator
ENTRYPOINT ["python", "-m", "tasks.video_generator.main"] 