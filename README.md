# Buch-AI-Server

## Tech stack

[![My Skills](https://skillicons.dev/icons?i=docker,fastapi,gcp,githubactions,py,terraform)](https://skillicons.dev)

## TL;DR — What's so cool about this repo anyways?

- **MCP / Agentic AI**: An MCP server that includes all the text and image generation tools used as part of the story generation workflow of Buch AI
- **Retrieval-augmented generation (RAG)**: Use of FAISS to store and retrieve story information for more consistent and grounded image generation
- **REST APIs**: A REST API backend server that includes text, image and video generation endpoints, authentication endpoints and database connectivity, which compliments a [separate React Native frontend](https://github.com/Buch-AI/Buch-AI-App)
- **DevOps**: A one-click push-to-main CI/CD pipeline trigger that builds and deploys Docker images and Cloud Run Jobs as part of an efficient DevOps setup using GitHub Actions and Terraform
- **Infrastructure design**: Use of Google Cloud Run Jobs that execute long-running tasks in parallel for better throughput and responsiveness
- **Software design**: Loosely-coupled and cohesive classes that: 1. adhere to object-oriented programming and software design best practices, 2. offer simple plug-and-play multi-vendor support (e.g. Hugging Face, Google Cloud, Llama API)

## Get started with development

1. Clone the repository.

```bash
git clone https://github.com/Buch-AI/Buch-AI-Server.git
```

2. Verify that you have a compatible Python version installed on your machine.
```bash
python --version
```

3. Install [uv](https://github.com/astral-sh/uv) (used as the package manager for this project).

4. Install the development dependencies.
```bash
cd Buch-AI-Server/
uv sync --all-groups
uv run pre-commit install
```

5. Run the API in development mode.
```bash
# Either, ...
uv run app/server/main.py
# Or, ...
uv run fastapi dev app/server/main.py --port 8080
```

## Deployment

For naming conventions, refer to https://stepan.wtf/cloud-naming-convention/.

### Continuous deployment

The GitHub repository has been configured to automatically deploy to Google Cloud Run upon pushing a new commit to the `main` branch. Refer to the GitHub Actions workflow manifest (`.github/workflows/deploy-google-cloud-run.yaml`) for more information.

### Manual step-by-step

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

## Google Cloud administration

### Setting IAM permissions and roles

```bash
export GOOGLE_CLOUD_PROJECT=$(gcloud config get-value core/project)
gcloud iam service-accounts create bai-buchai-p-svc-euwe2-gbq
gcloud iam service-accounts keys create ~/key.json --iam-account bai-buchai-p-svc-euwe2-gbq@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com
gcloud projects add-iam-policy-binding ${GOOGLE_CLOUD_PROJECT} --member "serviceAccount:bai-buchai-p-svc-euwe2-gbq@${GOOGLE_CLOUD_PROJECT}.iam.gserviceaccount.com" --role "roles/bigquery.user"
gcloud projects get-iam-policy $GOOGLE_CLOUD_PROJECT
```
