# Buch-AI-Server

## Get started with development

1. Clone the repository.

```bash
git clone https://github.com/Buch-AI/Buch-AI-Server.git
```

2. Verify that you have a compatible Python version installed on your machine.
```bash
python --version
```

3. Create a virtual environment and install the dependencies.
```bash
cd Buch-AI-Server/
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

4. Run the API in development mode.
```bash
fastapi dev main.py --port 8080
```

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