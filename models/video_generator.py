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
from pydantic import BaseModel, Field

from config import ASSETS_P_DIR


class VideoSlide(BaseModel):
    """Model representing a single slide in the video."""

    image: bytes = Field(..., description="Raw bytes of the image")
    captions: List[str] = Field(
        ..., min_items=1, description="List of captions to show sequentially"
    )


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
    SAMPLE_AUDIO_PATH = os.path.join(ASSETS_P_DIR, "sample-audio.mp3")

    @staticmethod
    def create_video_from_slides(slides: List[VideoSlide]) -> bytes:
        """
        Create a video from a list of VideoSlide objects, each containing an image and multiple captions.

        Args:
            slides: List of VideoSlide objects, each containing an image and its captions

        Returns:
            bytes: The generated video as bytes
        """
        # Video generation constants
        DURATION_PER_SLIDE = 5  # seconds
        AUDIO_VOLUME = 0.5
        FONT_SIZE = 18
        FONT_COLOR = "white"
        FONT_NAME = str(
            os.path.join(ASSETS_P_DIR, "Fredoka-VariableFont_wdth,wght.ttf")
        )
        CAPTION_BG_COLOR = "#000a"  # (R,G,B,A)
        VIDEO_FPS = 24
        VIDEO_CODEC = "libx264"
        AUDIO_CODEC = "aac"

        if not slides:
            raise ValueError("No slides provided")

        clips = []

        for slide in slides:
            # Convert image bytes to PIL Image
            img = Image.open(BytesIO(slide.image))

            # Create image clip
            img_clip = ImageClip(numpy.array(img))

            # Calculate duration for each caption
            caption_duration = DURATION_PER_SLIDE / len(slide.captions)

            # Create clips for each caption
            caption_clips = []
            for idx, caption in enumerate(slide.captions):
                # Create shadow text clip
                shadow_clip = TextClip(
                    caption,
                    fontsize=FONT_SIZE,
                    color="black",
                    method="caption",
                    size=(img.width, None),
                    font=FONT_NAME,
                )
                # Offset shadow slightly down and right
                shadow_clip = shadow_clip.set_position(
                    ("center", "bottom")
                ).set_opacity(0.8)
                shadow_clip = shadow_clip.margin(
                    top=2, left=2
                )  # Offset for shadow effect

                # Create main text clip
                text_clip = TextClip(
                    caption,
                    fontsize=FONT_SIZE,
                    color=FONT_COLOR,
                    bg_color=CAPTION_BG_COLOR,
                    method="caption",
                    size=(img.width, None),
                    font=FONT_NAME,
                )

                # Position text at bottom of image
                text_clip = text_clip.set_position(("center", "bottom"))

                # Set timing for both clips
                start_time = idx * caption_duration
                shadow_clip = shadow_clip.set_start(start_time).set_duration(
                    caption_duration
                )
                text_clip = text_clip.set_start(start_time).set_duration(
                    caption_duration
                )

                caption_clips.extend([shadow_clip, text_clip])

            # Combine image and all text clips
            composite = CompositeVideoClip([img_clip] + caption_clips)

            # Set duration for the entire slide
            composite = composite.set_duration(DURATION_PER_SLIDE)

            clips.append(composite)

        # Concatenate all clips
        final_clip = concatenate_videoclips(clips)
        total_duration = final_clip.duration

        # Load and prepare audio
        audio = AudioFileClip(VideoGenerator.SAMPLE_AUDIO_PATH)

        # Loop audio if video is longer than audio
        if total_duration > audio.duration:
            num_loops = int(total_duration / audio.duration) + 1
            audio = afx.audio_loop(audio, nloops=num_loops)

        # Set audio duration to match video
        audio = audio.set_duration(total_duration)

        # Set audio volume
        audio = audio.volumex(AUDIO_VOLUME)

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
                codec=VIDEO_CODEC,
                audio_codec=AUDIO_CODEC,
                fps=VIDEO_FPS,
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
