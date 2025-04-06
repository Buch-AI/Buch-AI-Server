import os

# Environment configuration
ENV = os.getenv("ENV", "p").lower()
if ENV not in ["d", "p"]:
    raise ValueError("ENV must be either 'd' (development) or 'p' (production)")

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Asset paths
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
ASSETS_D_DIR = os.path.join(ASSETS_DIR, "d")
ASSETS_P_DIR = os.path.join(ASSETS_DIR, "p")

# Google Cloud Storage URIs
GCLOUD_CREATIONS_STB_NAME = "bai-buchai-p-stb-usea1-creations"
