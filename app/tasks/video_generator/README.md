# Task: Video Generator

The Video Generator is implemented as a Cloud Run Job that processes story parts and images to create narrated videos. This README explains how the task works and how to interact with it.

## Architecture

The Video Generator operates in two modes:
- **d (Development)**: Runs locally and processes videos synchronously
- **p (Production)**: Runs as a Cloud Run Job for asynchronous processing

## Cloud Run Job Mechanism

1. **Job Trigger**
   - The task is triggered via the `/me/creation/{creation_id}/generate_video` endpoint
   - In production, this creates a new Cloud Run Job instance
   - The creation ID is passed as an argument to the task

2. **Job Execution**
   ```bash
   # Execute the task manually (for testing)
   gcloud run jobs execute buch-ai-video-generator --args <creation_id>

   # Example:
   gcloud run jobs execute buch-ai-video-generator --args 6d49a9c0-57ab-403d-af55-c02ad3dfcd70
   ```

3. **Processing Flow**
   - Task loads assets from Google Cloud Storage
   - Processes images and captions into a video
   - Uploads the final video back to Google Cloud Storage
   - Status can be checked via `/me/creation/{creation_id}/generate_video_status`

## Task Configuration

The task is configured with:
- Python 3.12 base image
- ImageMagick and FFmpeg for video processing
- Environment variables:
  - `ENV=p` for production mode
  - `MAGICK_HOME=/usr` for ImageMagick configuration
  - `PYTHONPATH=/app` for Python imports

## Storage Structure

Assets are stored in GCS with the following structure:
```
gs://bai-buchai-p-stb-usea1-creations/
└── {creation_id}/
    └── assets/
        ├── 1/
        │   ├── image.png
        │   └── 1.txt
        ├── 2/
        │   ├── image.png
        │   └── 1.txt
        └── video.mp4 (output)
```

## Error Handling

- Job failures are logged to Cloud Logging
- Status endpoint returns:
  - `pending`: Job is running or not started
  - `completed`: Video is available
  - `failed`: Job encountered an error

## Development Testing

For local testing, use the Video Generator demo script:
```bash
python -m app.scripts.tasks.video_generator_demo --assets-dir <path_to_assets>
```
