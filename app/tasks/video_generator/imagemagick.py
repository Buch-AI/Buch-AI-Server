import logging
import os
import platform
import subprocess
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_imagemagick_binary() -> Optional[str]:
    """
    Detect ImageMagick binary path across different environments.

    Returns:
        str: Path to ImageMagick binary or None if not found
    """
    # Check if MAGICK_HOME environment variable is set (Docker/production)
    magick_home = os.getenv("MAGICK_HOME")
    if magick_home:
        binary_path = os.path.join(magick_home, "bin", "convert")
        if os.path.exists(binary_path):
            return binary_path

    # Common paths based on OS
    if platform.system() == "Windows":
        common_paths = [
            r"C:\Program Files\ImageMagick-7.Q16\convert.exe",
            r"C:\Program Files\ImageMagick-7.Q16-HDRI\convert.exe",
        ]
    else:  # Unix-like systems (Linux, macOS)
        common_paths = [
            "/usr/bin/convert",
            "/usr/local/bin/convert",
            "/opt/homebrew/bin/convert",  # Common macOS Homebrew path
        ]

    # Check common paths
    for path in common_paths:
        if os.path.exists(path):
            return path

    # Try to find using which command on Unix-like systems
    try:
        if platform.system() != "Windows":
            result = subprocess.run(
                ["which", "convert"], capture_output=True, text=True, check=True
            )
            if result.stdout:
                return result.stdout.strip()
    except subprocess.SubprocessError:
        pass

    return None


def initialize_imagemagick() -> None:
    """
    Initialize ImageMagick for MoviePy usage.

    In MoviePy 2.x, configuration is done via environment variables.
    This function sets the IMAGEMAGICK_BINARY environment variable
    if ImageMagick is found on the system.

    Raises:
        RuntimeError: If ImageMagick binary is not found
    """
    # Check if IMAGEMAGICK_BINARY is already set
    if os.getenv("IMAGEMAGICK_BINARY"):
        logger.info(
            f"ImageMagick binary already configured: {os.getenv('IMAGEMAGICK_BINARY')}"
        )
        return

    imagemagick_binary = get_imagemagick_binary()
    if imagemagick_binary is None:
        logger.warning(
            "ImageMagick not found. Please install ImageMagick and ensure 'convert' binary is available in your system PATH. "
            "Some text effects may not work properly without ImageMagick."
        )
        return

    # Set environment variable for MoviePy 2.x
    os.environ["IMAGEMAGICK_BINARY"] = imagemagick_binary
    logger.info(f"ImageMagick binary configured: {imagemagick_binary}")
