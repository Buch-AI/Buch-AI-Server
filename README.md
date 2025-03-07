# Buch-AI-Server

## Deployment

For naming conventions, refer to https://stepan.wtf/cloud-naming-convention/.

1. Authenticate to Google Cloud.
```bash
gcloud auth login
```

2. Authenticate to the Docker repository on Google Artifact Registry.
```bash
gcloud auth configure-docker us-east1-docker.pkg.dev
```

3. Build the Docker image locally with the appropriate tag.
```bash
export TAG="us-east1-docker.pkg.dev/bai-buchai-p/bai-buchai-p-gar-usea1-docker/buch-ai-server:0.1.0"
docker build -t $TAG --platform linux/amd64 .
```

3. Push the image.
```bash
docker push $TAG
```

4. Deploy to Cloud Run.
```bash
gcloud run deploy bai-buchai-p-run-usea1-server --region us-east1 --image $TAG
```