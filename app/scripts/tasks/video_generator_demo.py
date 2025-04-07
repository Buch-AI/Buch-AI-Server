import argparse
import inspect
import logging
import os
from datetime import datetime
from typing import Any, Dict, get_type_hints

from app.tasks.video_generator.main import VideoGenerator
from config import PROJECT_ROOT

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure output directory exists
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_method_params() -> Dict[str, Dict[str, Any]]:
    """Get parameters and their default values from VideoGenerator.create_video_from_slides."""
    signature = inspect.signature(VideoGenerator.create_video_from_slides)
    type_hints = get_type_hints(VideoGenerator.create_video_from_slides)

    params = {}
    for name, param in signature.parameters.items():
        # Skip 'slides' as it's handled separately
        if name == "slides":
            continue

        param_type = type_hints.get(name, Any)
        default = param.default if param.default != inspect.Parameter.empty else None

        params[name] = {
            "type": param_type,
            "default": default,
            "help": f"Parameter '{name}' for video generation",
        }

    return params


def get_argparser() -> argparse.ArgumentParser:
    """Create argument parser with dynamically generated arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a video from images and caption files"
    )

    # Add required arguments
    parser.add_argument(
        "--assets-dir",
        type=str,
        required=True,
        help="Directory containing image files (.jpg/.png) and corresponding caption files (.txt)",
    )

    # Add dynamic arguments based on VideoGenerator parameters
    for name, param_info in get_method_params().items():
        arg_name = f"--{name.replace('_', '-')}"
        kwargs = {
            "type": param_info["type"],
            "default": param_info["default"],
            "help": param_info["help"],
        }
        parser.add_argument(arg_name, **kwargs)

    return parser


def main():
    parser = get_argparser()
    args = parser.parse_args()

    try:
        # Load assets using VideoGenerator's implementation
        logger.info("Loading assets...")
        slides = VideoGenerator.load_assets_from_directory(args.assets_dir)
        logger.info(f"Found {len(slides)} slides")

        # Generate video
        logger.info("Generating video...")
        video_bytes = VideoGenerator.create_video_from_slides(slides=slides)

        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"video_{timestamp}.mp4")

        # Save video to file
        with open(output_path, "wb") as f:
            f.write(video_bytes)

        logger.info(f"Video generated successfully: {output_path}")

    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
