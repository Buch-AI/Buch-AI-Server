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
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV ENV=p

# Copy only the requirements file first to leverage Docker cache
COPY pyproject.toml .
COPY setup.py .
RUN pip install --no-cache-dir -e .

# Copy the entire project
COPY . .

# Run the video generator
ENTRYPOINT ["python", "-m", "tasks.video_generator.main"] 