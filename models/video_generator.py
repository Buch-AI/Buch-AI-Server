import os
import platform
import subprocess
import tempfile
import uuid
from io import BytesIO
from typing import List, Optional

import numpy
from moviepy.config import change_settings
from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    afx,
    concatenate_videoclips,
)
from PIL import Image

from config import ASSETS_DIR


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


# Configure MoviePy to use ImageMagick
imagemagick_binary = get_imagemagick_binary()
if imagemagick_binary is None:
    raise RuntimeError(
        "ImageMagick not found. Please install ImageMagick and ensure "
        "'convert' binary is available in your system PATH"
    )

change_settings({"IMAGEMAGICK_BINARY": imagemagick_binary})


class VideoGenerator:
    """Handles video generation from images and text using MoviePy."""

    # Define audio path using the centrally configured assets directory
    AUDIO_PATH = os.path.join(ASSETS_DIR, "sample-audio.mp3")

    @staticmethod
    def create_video_from_assets(
        images: List[bytes],
        captions: List[str],
        duration_per_slide: int = 5,
        font_size: int = 30,
        font_color: str = "white",
        audio_volume: float = 0.5,
    ) -> bytes:
        """
        Create a video from a list of images and captions with background audio.

        Args:
            images: List of image bytes
            captions: List of text captions
            duration_per_slide: Duration for each slide in seconds
            font_size: Font size for captions
            font_color: Font color for captions
            audio_volume: Volume of background audio (0.0 to 1.0)

        Returns:
            bytes: The generated video as bytes
        """
        if len(images) != len(captions):
            raise ValueError("Number of images must match number of captions")

        clips = []

        for img_bytes, caption in zip(images, captions):
            # Convert image bytes to PIL Image
            img = Image.open(BytesIO(img_bytes))

            # Create image clip
            img_clip = ImageClip(numpy.array(img))

            # Create text clip
            txt_clip = TextClip(
                caption,
                fontsize=font_size,
                color=font_color,
                bg_color="black",
                method="caption",
                size=(img.width, None),
                font="FreeSans",
            )

            # Position text at bottom of image
            txt_clip = txt_clip.set_position(("center", "bottom"))

            # Combine image and text
            composite = CompositeVideoClip([img_clip, txt_clip])

            # Set duration
            composite = composite.set_duration(duration_per_slide)

            clips.append(composite)

        # Concatenate all clips
        final_clip = concatenate_videoclips(clips)
        total_duration = final_clip.duration

        # Load and prepare audio
        audio = AudioFileClip(VideoGenerator.AUDIO_PATH)

        # Loop audio if video is longer than audio
        if total_duration > audio.duration:
            num_loops = int(total_duration / audio.duration) + 1
            audio = afx.audio_loop(audio, nloops=num_loops)

        # Set audio duration to match video
        audio = audio.set_duration(total_duration)

        # Set audio volume
        audio = audio.volumex(audio_volume)

        # Add audio to final clip
        final_clip = final_clip.set_audio(audio)

        # Create a temporary file
        temp_output_path = os.path.join(
            tempfile.gettempdir(), f"video_{uuid.uuid4()}.mp4"
        )

        try:
            # Write to temporary file
            final_clip.write_videofile(
                temp_output_path,
                codec="libx264",
                audio_codec="aac",  # Use AAC codec for audio
                fps=24,
            )

            # Read the temporary file into bytes
            with open(temp_output_path, "rb") as f:
                video_bytes = f.read()

            # Close clips to free up resources
            audio.close()
            final_clip.close()
            for clip in clips:
                clip.close()

            return video_bytes

        finally:
            # Clean up temporary file
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
