import argparse
import inspect
import os
from datetime import datetime
from typing import Any, Dict, List, get_type_hints

from config import PROJECT_ROOT
from models.video_generator import VideoGenerator, VideoSlide

# Ensure output directory exists
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_assets_from_directory(assets_dir: str) -> List[VideoSlide]:
    """Load images and their corresponding captions from a directory.

    Expected directory structure:
    assets_dir/
        ├── 1/
        │   ├── image.jpg
        │   ├── 1.txt
        │   ├── 2.txt
        │   └── ...
        ├── 2/
        │   ├── image.jpg
        │   ├── 1.txt
        │   ├── 2.txt
        │   └── ...
        └── ...
    """
    # Get all subdirectories (they should be numbered)
    subdirs = sorted(
        [
            d
            for d in os.listdir(assets_dir)
            if os.path.isdir(os.path.join(assets_dir, d))
        ]
    )

    if not subdirs:
        raise ValueError(f"No slide directories found in {assets_dir}")

    slides = []

    for subdir in subdirs:
        subdir_path = os.path.join(assets_dir, subdir)

        # Find the image file (should be named image.jpg/png/etc)
        image_files = [
            f
            for f in os.listdir(subdir_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
            and os.path.splitext(f.lower())[0] == "image"
        ]

        if not image_files:
            raise ValueError(f"No image file found in {subdir_path}")

        if len(image_files) > 1:
            raise ValueError(f"Multiple image files found in {subdir_path}")

        # Load image
        img_path = os.path.join(subdir_path, image_files[0])
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        # Get all caption files (should be numbered .txt files)
        caption_files = sorted(
            [f for f in os.listdir(subdir_path) if f.lower().endswith(".txt")]
        )

        if not caption_files:
            raise ValueError(f"No caption files found in {subdir_path}")

        # Load captions
        captions = []
        for txt_file in caption_files:
            txt_path = os.path.join(subdir_path, txt_file)
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
                if not caption:
                    raise ValueError(f"Empty caption file: {txt_path}")
                captions.append(caption)

        # Create VideoSlide
        slide = VideoSlide(image=image_bytes, captions=captions)
        slides.append(slide)

    return slides


def get_method_params() -> Dict[str, Dict[str, Any]]:
    """Get parameters and their default values from VideoGenerator.create_video_from_slides."""
    signature = inspect.signature(VideoGenerator.create_video_from_slides)
    type_hints = get_type_hints(VideoGenerator.create_video_from_slides)

    params = {}
    for name, param in signature.parameters.items():
        # Skip 'slides' and 'cls' as they're handled separately
        if name in ("slides", "cls"):
            continue

        param_type = type_hints.get(name, Any)
        default = param.default if param.default != inspect.Parameter.empty else None

        params[name] = {
            "type": param_type,
            "default": default,
            "help": f"Parameter '{name}' for video generation",
        }

    return params


def setup_argparser() -> argparse.ArgumentParser:
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
    parser = setup_argparser()
    args = parser.parse_args()

    try:
        # Load assets
        print("Loading assets...")
        slides = load_assets_from_directory(args.assets_dir)
        print(f"Found {len(slides)} slides")

        # Build kwargs for VideoGenerator dynamically
        kwargs = {"slides": slides}

        # Add other parameters from args, excluding special handling args
        special_args = {"assets_dir"}
        for key, value in vars(args).items():
            if key not in special_args and value is not None:
                kwargs[key] = value

        # Generate video
        print("Generating video...")
        video_bytes = VideoGenerator.create_video_from_slides(**kwargs)

        # Create output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"video_{timestamp}.mp4")

        # Save video to file
        with open(output_path, "wb") as f:
            f.write(video_bytes)

        print(f"Video generated successfully: {output_path}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
