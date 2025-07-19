import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment configuration
ENV = os.getenv("BUCHAI_ENV", "p").lower()
if ENV not in ["d", "p"]:
    raise ValueError("BUCHAI_ENV must be either 'd' (development) or 'p' (production)")

# API Keys
AUTH_JWT_KEY = os.getenv("BUCHAI_AUTH_JWT_KEY")
if not AUTH_JWT_KEY:
    raise ValueError("BUCHAI_AUTH_JWT_KEY environment variable is not set")

HF_API_KEY = os.getenv("BUCHAI_HF_API_KEY")
if not HF_API_KEY:
    raise ValueError("BUCHAI_HF_API_KEY environment variable is not set")

# NOTE: IPinfo.io API key is optional - free tier allows limited usage without token
IPINFO_API_KEY = os.getenv("BUCHAI_IPINFO_API_KEY")

# Google Cloud Storage URIs
GCLOUD_STB_CREATIONS_NAME = "bai-buchai-p-stb-usea1-creations"

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Asset paths
ASSETS_DIR = os.path.join(PROJECT_ROOT, "app", "assets")
ASSETS_D_DIR = os.path.join(ASSETS_DIR, "d")
ASSETS_P_DIR = os.path.join(ASSETS_DIR, "p")

# Output directory for development environment
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
