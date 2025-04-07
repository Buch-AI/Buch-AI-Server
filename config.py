import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment configuration
ENV = os.getenv("ENV", "p").lower()
if ENV not in ["d", "p"]:
    raise ValueError("ENV must be either 'd' (development) or 'p' (production)")

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Asset paths
ASSETS_DIR = os.path.join(PROJECT_ROOT, "app", "assets")
ASSETS_D_DIR = os.path.join(ASSETS_DIR, "d")
ASSETS_P_DIR = os.path.join(ASSETS_DIR, "p")

# Google Cloud Storage URIs
GCLOUD_STB_CREATIONS_NAME = "bai-buchai-p-stb-usea1-creations"

# API Keys
AUTH_JWT_KEY = os.getenv("AUTH_JWT_KEY")
if not AUTH_JWT_KEY:
    raise ValueError("AUTH_JWT_KEY environment variable is not set")

HF_API_KEY = os.getenv("HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("HF_API_KEY environment variable is not set")
