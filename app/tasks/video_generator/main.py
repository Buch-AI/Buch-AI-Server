import logging
import os
import platform
import subprocess
import tempfile
import uuid
from io import BytesIO
from typing import List, Optional
from urllib.parse import urlparse

import numpy
from google.cloud import storage
from moviepy.config import change_settings
from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    afx,
    concatenate_videoclips,
)
from moviepy.video.fx.fadein import fadein
from moviepy.video.fx.fadeout import fadeout
from PIL import Image
from pydantic import BaseModel, Field

from config import ASSETS_P_DIR, GCLOUD_STB_CREATIONS_NAME

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


# Configure MoviePy to use ImageMagick
imagemagick_binary = get_imagemagick_binary()
if imagemagick_binary is None:
    raise RuntimeError(
        "ImageMagick not found. Please install ImageMagick and ensure "
        "'convert' binary is available in your system PATH"
    )

change_settings({"IMAGEMAGICK_BINARY": imagemagick_binary})


class VideoSlide(BaseModel):
    """Model representing a single slide in the video."""

    image: bytes = Field(..., description="Raw bytes of the image")
    captions: List[str] = Field(
        ..., min_items=1, description="List of captions to show sequentially"
    )


class VideoGenerator:
    """Handles video generation from images and text using MoviePy."""

    # Define audio path using the centrally configured assets directory
    SAMPLE_AUDIO_PATH = os.path.join(ASSETS_P_DIR, "sample-audio.mp3")

    @staticmethod
    def _is_gcs_path(path: str) -> bool:
        """Check if a path is a Google Cloud Storage URL."""
        return path.startswith("gs://")

    @staticmethod
    def _parse_gcs_path(gcs_path: str) -> tuple[str, str]:
        """Parse a GCS path into bucket and blob path."""
        parsed = urlparse(gcs_path)
        if not parsed.netloc:
            raise ValueError(f"Invalid GCS path: {gcs_path}")
        return parsed.netloc, parsed.path.lstrip("/")

    @staticmethod
    def _read_file_from_gcs(bucket_name: str, blob_path: str) -> bytes:
        """Read a file from Google Cloud Storage."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.download_as_bytes()

    @staticmethod
    def _list_gcs_directory(bucket_name: str, prefix: str) -> List[str]:
        """List contents of a GCS directory."""
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        return [blob.name for blob in blobs]

    @classmethod
    def load_assets_from_directory(cls, directory_path: str) -> List[VideoSlide]:
        """Load images and their corresponding captions from a directory.

        Supports both local filesystem and Google Cloud Storage paths (gs://).

        Expected directory structure (both local and GCS):
        directory/
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

        Args:
            directory_path: Local path or GCS URL (gs://) to the assets directory

        Returns:
            List[VideoSlide]: List of VideoSlide objects containing images and captions

        Raises:
            ValueError: If directory structure is invalid or required files are missing
        """
        is_gcs = cls._is_gcs_path(directory_path)
        slides = []

        if is_gcs:
            bucket_name, prefix = cls._parse_gcs_path(directory_path)
            all_paths = cls._list_gcs_directory(bucket_name, prefix)

            # Group files by slide number
            slide_files = {}
            for path in all_paths:
                # Extract slide number from path
                rel_path = path[len(prefix) :].lstrip("/")
                parts = rel_path.split("/")
                if len(parts) != 2:  # Should be "slide_num/filename"
                    continue

                slide_num, filename = parts
                if slide_num not in slide_files:
                    slide_files[slide_num] = []
                slide_files[slide_num].append(filename)

            # Process each slide
            for slide_num in sorted(slide_files.keys()):
                files = slide_files[slide_num]

                # Find image file
                image_files = [
                    f
                    for f in files
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))
                    and os.path.splitext(f.lower())[0] == "image"
                ]

                if not image_files:
                    raise ValueError(f"No image file found in slide {slide_num}")
                if len(image_files) > 1:
                    raise ValueError(f"Multiple image files found in slide {slide_num}")

                # Load image
                image_path = f"{prefix.rstrip('/')}/{slide_num}/{image_files[0]}"
                image_bytes = cls._read_file_from_gcs(bucket_name, image_path)

                # Load captions
                caption_files = sorted([f for f in files if f.lower().endswith(".txt")])
                if not caption_files:
                    raise ValueError(f"No caption files found in slide {slide_num}")

                captions = []
                for txt_file in caption_files:
                    txt_path = f"{prefix.rstrip('/')}/{slide_num}/{txt_file}"
                    caption_bytes = cls._read_file_from_gcs(bucket_name, txt_path)
                    caption = caption_bytes.decode("utf-8").strip()
                    if not caption:
                        raise ValueError(f"Empty caption file: {txt_path}")
                    captions.append(caption)

                slides.append(VideoSlide(image=image_bytes, captions=captions))

        else:
            # Handle local filesystem
            subdirs = sorted(
                [
                    d
                    for d in os.listdir(directory_path)
                    if os.path.isdir(os.path.join(directory_path, d))
                ]
            )

            if not subdirs:
                raise ValueError(f"No slide directories found in {directory_path}")

            for subdir in subdirs:
                subdir_path = os.path.join(directory_path, subdir)

                # Find image file
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

                # Load captions
                caption_files = sorted(
                    [f for f in os.listdir(subdir_path) if f.lower().endswith(".txt")]
                )

                if not caption_files:
                    raise ValueError(f"No caption files found in {subdir_path}")

                captions = []
                for txt_file in caption_files:
                    txt_path = os.path.join(subdir_path, txt_file)
                    with open(txt_path, "r", encoding="utf-8") as f:
                        caption = f.read().strip()
                        if not caption:
                            raise ValueError(f"Empty caption file: {txt_path}")
                        captions.append(caption)

                slides.append(VideoSlide(image=image_bytes, captions=captions))

        return slides

    @staticmethod
    def get_caption_duration(text: str, wpm: int) -> float:
        """
        Calculate the duration needed to read a caption at a given words per minute rate.

        Args:
            text: The caption text
            wpm: Words per minute reading speed

        Returns:
            float: Duration in seconds needed to read the text
        """
        # Count words (split by whitespace)
        word_count = len(text.split())

        # Calculate base duration (words / (words per minute / 60 seconds))
        base_duration = (word_count / wpm) * 60

        # Add a minimum buffer time of 1.5 seconds for visual comprehension
        return max(base_duration + 1.5, 2.0)  # Ensure at least 2 seconds per caption

    @staticmethod
    def get_gradient_background(
        width: int, height: int, alpha_bottom: float = 0.7, alpha_top: float = 0.0
    ) -> numpy.ndarray:
        """
        Create a gradient background that fades from black to transparent vertically.

        Args:
            width: Width of the background
            height: Height of the background
            alpha_bottom: Starting alpha value at the bottom (0-1)
            alpha_top: Ending alpha value at the top (0-1)

        Returns:
            numpy.ndarray: RGBA array representing the gradient background
        """
        # Create a vertical gradient of alpha values (reversed to fade upward)
        alpha_gradient = numpy.linspace(alpha_top, alpha_bottom, height)[
            :, numpy.newaxis
        ]

        # Create the RGBA array (black background with varying alpha)
        gradient = numpy.zeros((height, width, 4))
        # Set RGB to black (0, 0, 0)
        gradient[:, :, :3] = 0
        # Set alpha channel with gradient and scale to 0-255 range
        gradient[:, :, 3] = numpy.tile(alpha_gradient, (1, width)) * 255

        # Convert to uint8 for proper image handling
        return gradient.astype(numpy.uint8)

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
        AUDIO_VOLUME = 0.5
        WORDS_PER_MIN = 220
        FONT_SIZE = 18
        FONT_COLOR = "white"
        FONT_NAME = str(
            os.path.join(ASSETS_P_DIR, "Fredoka-VariableFont_wdth,wght.ttf")
        )
        CAPTION_PADDING = 40  # Padding from the bottom and sides of the frame
        VIDEO_FPS = 24
        VIDEO_CODEC = "libx264"
        AUDIO_CODEC = "aac"
        FADE_DURATION = 0.5  # Duration of fade transitions in seconds

        if not slides:
            raise ValueError("No slides provided")

        slide_clips = []

        for slide in slides:
            # Convert image bytes to PIL Image
            image = Image.open(BytesIO(slide.image))

            # Create image clip
            image_clip = ImageClip(numpy.array(image))

            # Calculate durations for each caption based on text content
            caption_durations = [
                VideoGenerator.get_caption_duration(caption, WORDS_PER_MIN)
                for caption in slide.captions
            ]

            caption_clips = []
            current_time = 0

            for idx, (caption, duration) in enumerate(
                zip(slide.captions, caption_durations)
            ):
                # Create main text clip first to get its size
                text_clip = TextClip(
                    caption,
                    fontsize=FONT_SIZE,
                    color=FONT_COLOR,
                    method="caption",
                    size=(
                        image.width - 2 * CAPTION_PADDING,
                        None,
                    ),  # Add padding on sides
                    font=FONT_NAME,
                )

                # Get text clip size and calculate positions
                text_height = text_clip.size[1]
                text_y_pos = (
                    image.height - text_height - CAPTION_PADDING
                )  # Add bottom padding
                gradient_height = int(
                    text_height * 2
                )  # Make gradient taller for more fade

                # Create gradient background
                gradient = VideoGenerator.get_gradient_background(
                    width=image.width,
                    height=gradient_height,
                    alpha_bottom=1.0,  # Fully opaque at bottom
                    alpha_top=0.0,  # Fully transparent at top
                )

                # Create gradient clip
                gradient_clip = ImageClip(gradient)

                # Create shadow text clip
                shadow_clip = TextClip(
                    caption,
                    fontsize=FONT_SIZE,
                    color="black",
                    method="caption",
                    size=(
                        image.width - 2 * CAPTION_PADDING,
                        None,
                    ),  # Add padding on sides
                    font=FONT_NAME,
                )

                # Position clips
                gradient_y_pos = image.height - gradient_height
                gradient_clip = gradient_clip.set_position(("center", gradient_y_pos))
                shadow_clip = shadow_clip.set_position(
                    ("center", text_y_pos + 2)
                ).set_opacity(0.8)
                text_clip = text_clip.set_position(("center", text_y_pos))

                # Set timing for all clips using calculated duration
                gradient_clip = gradient_clip.set_duration(duration)
                shadow_clip = shadow_clip.set_duration(duration)
                text_clip = text_clip.set_duration(duration)

                # Add fade transitions
                if (
                    duration > 2 * FADE_DURATION
                ):  # Only add fades if clip is long enough
                    gradient_clip = gradient_clip.fx(fadein, FADE_DURATION).fx(
                        fadeout, FADE_DURATION
                    )
                    shadow_clip = shadow_clip.fx(fadein, FADE_DURATION).fx(
                        fadeout, FADE_DURATION
                    )
                    text_clip = text_clip.fx(fadein, FADE_DURATION).fx(
                        fadeout, FADE_DURATION
                    )

                # Set start times
                gradient_clip = gradient_clip.set_start(current_time)
                shadow_clip = shadow_clip.set_start(current_time)
                text_clip = text_clip.set_start(current_time)

                current_time += duration

                # Add clips in order: gradient (bottom), shadow (middle), text (top)
                caption_clips.extend([gradient_clip, shadow_clip, text_clip])

            # Create composite of image and all captions
            slide_duration = sum(caption_durations)
            image_clip = image_clip.set_duration(slide_duration)
            slide_composite = CompositeVideoClip([image_clip] + caption_clips)

            # Add fade transitions between slides
            if slide_duration > 2 * FADE_DURATION:
                slide_composite = slide_composite.set_duration(
                    slide_duration
                )  # Ensure duration is set
                slide_composite = slide_composite.fx(fadein, FADE_DURATION).fx(
                    fadeout, FADE_DURATION
                )

            slide_clips.append(slide_composite)

        # Concatenate all slides
        final_clip = concatenate_videoclips(slide_clips, method="compose")
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
            for clip in slide_clips:
                clip.close()

            return video_bytes

        finally:
            # Clean up temporary file
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)


if __name__ == "__main__":
    # This is the entry point when running as a Cloud Run Job
    import sys

    if len(sys.argv) != 2:
        logger.error("Usage: python main.py <creation_id>")
        sys.exit(1)

    creation_id = sys.argv[1]

    try:
        # Load slides from GCS
        gcs_path = f"gs://{GCLOUD_STB_CREATIONS_NAME}/{creation_id}/assets"
        slides = VideoGenerator.load_assets_from_directory(gcs_path)

        if not slides:
            raise ValueError(f"No valid slides found for creation {creation_id}")

        # Generate video
        video_bytes = VideoGenerator.create_video_from_slides(slides=slides)

        # Upload to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCLOUD_STB_CREATIONS_NAME)
        video_blob_path = f"{creation_id}/assets/video.mp4"
        video_blob = bucket.blob(video_blob_path)
        video_blob.upload_from_string(video_bytes, content_type="video/mp4")

    except Exception as e:
        logger.error(
            f"Error generating video for creation {creation_id}: {str(e)}",
            exc_info=True,
        )
        raise
