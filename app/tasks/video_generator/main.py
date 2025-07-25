"""
Whoever fixes this behemoth of a monolith is a legend.
1000 lines of code go brrr.
Also, I think I just witnessed Cursor autocompleting the meme above. AI is insane.
"""

import asyncio
import concurrent.futures
import logging
import os
import sys
import tempfile
import textwrap
import uuid
from functools import lru_cache, partial
from io import BytesIO
from typing import List, Literal, Optional
from urllib.parse import urlparse

import numpy
from google.cloud import storage
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoClip,
    concatenate_videoclips,
)
from moviepy.video.fx.FadeIn import FadeIn
from moviepy.video.fx.FadeOut import FadeOut
from PIL import Image, ImageFont

from app.tasks.video_generator.dubber import GoogleCloudDubber
from app.tasks.video_generator.imagemagick import initialize_imagemagick
from app.tasks.video_generator.video_slide import VideoSlide
from config import ASSETS_P_DIR, ENV, GCLOUD_STB_CREATIONS_NAME, OUTPUT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize ImageMagick
initialize_imagemagick()


class VideoGenerator:
    """Handles video generation from images and text using MoviePy."""

    # NOTE: Google Cloud Storage connectivity

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

    # NOTE: Utility functions for optimized video generation

    @staticmethod
    def _resize_image(image_bytes: bytes, target_width: int) -> numpy.ndarray:
        """
        Optimize image size and format for video processing.

        Args:
            image_bytes: Raw image bytes
            target_width: Target width for optimization

        Returns:
            numpy.ndarray: Optimized image array
        """
        image = Image.open(BytesIO(image_bytes))

        # Calculate optimal size maintaining aspect ratio
        aspect_ratio = image.height / image.width
        new_height = int(target_width * aspect_ratio)

        # Resize with high-quality resampling
        resized = image.resize((target_width, new_height), Image.LANCZOS)

        # Convert to RGB if needed (removes alpha channel, reduces processing)
        if resized.mode != "RGB":
            resized = resized.convert("RGB")

        return numpy.array(resized)

    @staticmethod
    def _save_clip(
        clip, output_path: str, fps: int, video_codec: str, audio_codec: str
    ) -> None:
        """
        Write video with optimized encoding settings.

        Args:
            clip: MoviePy clip to write
            output_path: Output file path
            fps: Target frame rate
        """
        clip.write_videofile(
            output_path,
            codec=video_codec,
            audio_codec=audio_codec,
            fps=fps,
            preset="fast",  # Balanced speed vs quality
            ffmpeg_params=[
                "-crf",
                "23",  # Constant Rate Factor (good quality/size balance)
                "-threads",
                "0",  # Use all available CPU cores
                "-movflags",
                "+faststart",  # Optimize for streaming
                "-pix_fmt",
                "yuv420p",  # Ensure compatibility
            ],
        )

    @staticmethod
    def _get_slide_batch(slides: List[VideoSlide], batch_size: int):
        """
        Process slides in smaller batches to manage memory usage.

        Args:
            slides: List of video slides
            batch_size: Number of slides to process at once

        Yields:
            List[VideoSlide]: Batches of slides
        """
        for i in range(0, len(slides), batch_size):
            yield slides[i : i + batch_size]

    # NOTE: Video effects

    @staticmethod
    def _get_gradient_background(
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
    def _create_zoom_func(
        zoom_factor: float,
        duration: float,
        zoom_style: Literal["linear", "ease_in", "ease_out", "ease_in_out"],
        video_width: int,
        video_height: int,
    ):
        """
        Create a centered zoom function that zooms around the center point of the image.

        Args:
            zoom_factor: Final zoom factor (1.0 = no zoom, 1.2 = 20% zoom)
            duration: Total duration of the zoom effect in seconds
            zoom_style: Speed curve - "linear", "ease_in", "ease_out", or "ease_in_out"
            video_width: Width of the video frame
            video_height: Height of the video frame

        Returns:
            Function that takes time (t) and returns a resize function for centered zoom
        """

        def zoom_func(t):
            # Normalize time to 0-1 range
            progress = min(t / duration, 1.0)

            # Apply easing curves
            if zoom_style == "ease_in":
                # Quadratic ease in (slow start, fast end)
                progress = progress**2
            elif zoom_style == "ease_out":
                # Quadratic ease out (fast start, slow end)
                progress = 1 - (1 - progress) ** 2
            elif zoom_style == "ease_in_out":
                # Quadratic ease in-out (slow start and end, fast middle)
                if progress < 0.5:
                    progress = 2 * progress**2
                else:
                    progress = 1 - 2 * (1 - progress) ** 2
            # "linear" or default case - no modification needed

            # Interpolate between 1.0 (start) and zoom_factor (end)
            current_zoom = 1.0 + (zoom_factor - 1.0) * progress

            # Calculate new dimensions
            new_width = int(video_width * current_zoom)
            new_height = int(video_height * current_zoom)

            return (new_width, new_height)

        return zoom_func

    @staticmethod
    def _apply_zoom_to_clip(
        image_clip,
        zoom_factor: float,
        duration: float,
        zoom_style: str,
        video_width: int,
        video_height: int,
    ):
        """
        Apply a centered zoom effect to an image clip.

        Args:
            image_clip: The image clip to zoom
            zoom_factor: Final zoom factor
            duration: Duration of zoom
            zoom_style: Zoom easing style
            video_width: Target video width
            video_height: Target video height

        Returns:
            ImageClip with centered zoom effect applied
        """
        zoom_func = VideoGenerator._create_zoom_func(
            zoom_factor, duration, zoom_style, video_width, video_height
        )

        def make_frame(t):
            # Get the current zoom dimensions
            new_width, new_height = zoom_func(t)

            # Get the original frame
            frame = image_clip.get_frame(t)
            original_height, original_width = frame.shape[:2]

            # Resize the frame to the new dimensions
            from PIL import Image as PILImage

            pil_image = PILImage.fromarray(frame)
            resized_image = pil_image.resize((new_width, new_height), PILImage.LANCZOS)
            resized_frame = numpy.array(resized_image)

            # Create output frame with original dimensions
            output_frame = numpy.zeros(
                (video_height, video_width, 3), dtype=numpy.uint8
            )

            # Calculate center position for cropping/positioning
            center_x = new_width // 2
            center_y = new_height // 2

            # Calculate crop boundaries to center the zoom
            crop_left = max(0, center_x - video_width // 2)
            crop_right = min(new_width, center_x + video_width // 2)
            crop_top = max(0, center_y - video_height // 2)
            crop_bottom = min(new_height, center_y + video_height // 2)

            # Extract the centered crop
            cropped_frame = resized_frame[crop_top:crop_bottom, crop_left:crop_right]

            # Calculate position in output frame
            paste_x = (video_width - cropped_frame.shape[1]) // 2
            paste_y = (video_height - cropped_frame.shape[0]) // 2

            # Place the cropped frame in the center of the output
            end_y = paste_y + cropped_frame.shape[0]
            end_x = paste_x + cropped_frame.shape[1]

            output_frame[paste_y:end_y, paste_x:end_x] = cropped_frame

            return output_frame

        # Create a new clip with the zoom effect
        zoomed_clip = VideoClip(make_frame, duration=duration)
        zoomed_clip.fps = image_clip.fps if hasattr(image_clip, "fps") else 24

        return zoomed_clip

    # NOTE: Captions and text

    @staticmethod
    def _calculate_caption_duration(text: str, wpm: int) -> float:
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
    def _calculate_chars_per_line(
        font_path: str, font_size: int, max_width: int
    ) -> int:
        """
        Calculate the number of characters that can fit on a line based on actual font metrics.

        Args:
            font_path: Path to the font file
            font_size: Font size in pixels
            max_width: Maximum width available for text

        Returns:
            int: Number of characters that can fit on a line
        """
        # Load the font
        font = ImageFont.truetype(font_path, font_size)

        # Sample text to measure average character width
        # Using a mix of common characters including narrow (i, l) and wide (m, w) characters
        sample_text = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?"
        )

        # Get the bounding box of the sample text
        bbox = font.getbbox(sample_text)
        text_width = bbox[2] - bbox[0]  # width = right - left

        # Calculate average character width
        avg_char_width = text_width / len(sample_text)

        # Calculate characters per line with a small safety margin (0.95 factor)
        chars_per_line = int(max_width / avg_char_width)

        # Ensure minimum characters per line
        return chars_per_line

    @staticmethod
    @lru_cache(maxsize=256)  # Increased cache size for wrapped text clips
    def _create_wrapped_text_clip(
        text: str,
        font_size: int,
        color: str,
        font_path: str,
        max_width: int,
        line_spacing: float = 1.2,
    ) -> TextClip:
        """
        Create a text clip with proper word wrapping and no word breaking.
        Now cached to avoid recreating identical clips.

        Args:
            text: Text content
            font_size: Font size
            color: Text color
            font_path: Path to font file
            max_width: Maximum width for text (accounts for padding)
            line_spacing: Line spacing multiplier

        Returns:
            TextClip: Text clip with proper word wrapping
        """

        if ENV == "d":
            # Clear the old cached function to ensure fresh rendering
            VideoGenerator._create_wrapped_text_clip.cache_clear()

        # Use accurate font-based calculation instead of rough approximation
        chars_per_line = VideoGenerator._calculate_chars_per_line(
            font_path, font_size, max_width
        )

        # Wrap text without breaking words
        wrapped_lines = textwrap.fill(
            text,
            width=chars_per_line,
            break_long_words=False,  # Don't break long words
            break_on_hyphens=False,  # Don't break on hyphens
        ).split("\n")

        # Join lines back with proper spacing and add extra newline for bottom padding
        wrapped_text = (
            "\n".join(wrapped_lines) + "\n"
        )  # Add newline at the end for bottom spacing

        # Create the text clip with proper word wrapping - size required for caption method
        text_clip = TextClip(
            text=wrapped_text,
            font_size=font_size,
            color=color,
            method="caption",
            font=font_path,
            size=(max_width, None),  # Size required for caption method
            text_align="center",
            interline=int(font_size * (line_spacing - 1)),  # Line spacing
        )

        return text_clip

    @staticmethod
    async def _create_wrapped_text_clips_batch(
        texts_and_colors: List[tuple[str, str]],
        font_size: int,
        font_path: str,
        max_width: int,
        line_spacing: float = 1.2,
    ) -> List[TextClip]:
        """
        Create multiple wrapped text clips in parallel for better performance.
        Uses the cached _create_wrapped_text_clip function internally.

        Args:
            texts_and_colors: List of (text, color) tuples
            font_size: Font size for text
            font_path: Path to font file
            max_width: Maximum width for text clips
            line_spacing: Line spacing multiplier

        Returns:
            List[TextClip]: List of created wrapped text clips
        """
        loop = asyncio.get_event_loop()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            # Create partial function for text clip creation
            create_wrapped_clip_func = partial(
                VideoGenerator._create_wrapped_text_clip,
                font_size=font_size,
                font_path=font_path,
                max_width=max_width,
                line_spacing=line_spacing,
            )

            # Submit all text clip creation tasks
            tasks = [
                loop.run_in_executor(
                    executor,
                    lambda text_color: create_wrapped_clip_func(
                        text=text_color[0], color=text_color[1]
                    ),
                    text_color,
                )
                for text_color in texts_and_colors
            ]

            # Wait for all tasks to complete
            text_clips = await asyncio.gather(*tasks)

        return text_clips

    # NOTE: Main function

    @staticmethod
    async def create_video_from_slides(
        slides: List[VideoSlide], cost_centre_id: Optional[str] = None
    ) -> bytes:
        """
        Create a video from a list of VideoSlide objects, each containing an image and multiple captions.

        Args:
            slides: List of VideoSlide objects, each containing an image and its captions
            cost_centre_id: Optional cost centre ID for tracking TTS usage costs

        Returns:
            bytes: The generated video as bytes

        NOTE:
            This function includes a zoom-in effect on images. The zoom settings are configured
            as constants within the function and can be modified in the source code if needed.
        """
        logger.info(f"Starting video generation from {len(slides)} slides")

        # Video dimensions
        VIDEO_WIDTH = 480
        VIDEO_HEIGHT = 720

        # Video generation constants
        VIDEO_FPS = 20
        VIDEO_CODEC = "libx264"
        AUDIO_CODEC = "aac"

        # Fade effect constants
        FADE_DURATION = 0.4

        # Zoom effect constants
        ENABLE_ZOOM = False  # Whether to enable the zoom-in effect on images
        ZOOM_FACTOR = 1.08
        ZOOM_STYLE = "linear"  # Speed curve for zoom effect - "linear", "ease_in", "ease_out", or "ease_in_out"

        # Dubbing constants
        ENABLE_DUBBING = True  # Whether to enable text-to-speech audio generation
        DUBBING_WPM = 240

        # Caption constants
        FONT_SIZE = 20
        FONT_COLOR = "white"
        FONT_NAME = str(os.path.join(ASSETS_P_DIR, "Helvetica-Bold.ttf"))
        CAPTION_PADDING = 5  # Horizontal padding from left and right edges (controls text width and margins)

        # Performance constants
        SLIDE_BATCH_SIZE = 4  # Process slides in batches to manage memory

        if not slides:
            raise ValueError("No slides provided")

        # Initialize the dubber and generate audio if dubbing is enabled
        if ENABLE_DUBBING:
            logger.info("Initializing Google Cloud Dubber")
            dubber = GoogleCloudDubber(
                language_code="en-GB",
                voice_name="en-GB-Standard-C",
                speaking_rate=1.0,
                pitch=0.0,
                cost_centre_id=cost_centre_id,
            )

            # Generate audio for all slides
            logger.info("Generating audio for all slides")
            _slides = await dubber.create_audio_from_slides(slides)
            logger.info(f"Audio generation complete for {len(_slides)} slides")

            # Verify audio was generated
            total_audio_clips = sum(
                len(slide.caption_dubs) if slide.caption_dubs else 0
                for slide in _slides
            )
            total_captions = sum(len(slide.captions) for slide in _slides)
            logger.info(
                f"Generated {total_audio_clips} audio clips for {total_captions} total captions"
            )

            if total_audio_clips == 0:
                logger.error(
                    "No audio clips were generated! Check Google Cloud TTS configuration and credentials."
                )
            elif total_audio_clips != total_captions:
                logger.warning(
                    f"Audio clip count mismatch: {total_audio_clips} clips vs. {total_captions} captions"
                )
        else:
            logger.info("Dubbing disabled - using original slides without audio")
            _slides = slides

        slide_clips = []

        # Process slides in batches for better memory management
        for batch_slides in VideoGenerator._get_slide_batch(_slides, SLIDE_BATCH_SIZE):
            batch_clips = []

            for slide_idx, slide in enumerate(batch_slides, 1):
                # Calculate global slide index
                global_slide_idx = len(slide_clips) + slide_idx
                logger.info(f"Processing slide {global_slide_idx}/{len(_slides)}...")

                # Verify slide has captions and audio (only if dubbing is enabled)
                if ENABLE_DUBBING:
                    if not hasattr(slide, "caption_dubs") or slide.caption_dubs is None:
                        logger.error(
                            f"Slide {global_slide_idx} has no caption_dubs attribute"
                        )
                        raise ValueError(
                            f"Slide {global_slide_idx} is missing audio data"
                        )

                    if len(slide.captions) != len(slide.caption_dubs):
                        logger.error(
                            f"Slide {global_slide_idx} has {len(slide.captions)} captions but {len(slide.caption_dubs)} audio clips"
                        )
                        raise ValueError(
                            f"Caption and audio count mismatch for slide {global_slide_idx}"
                        )

                resized_image = VideoGenerator._resize_image(slide.image, VIDEO_WIDTH)
                logger.info(
                    f"Slide {global_slide_idx} image resized to {resized_image.shape}"
                )

                # Create a blank black canvas with the target dimensions
                canvas_height, canvas_width = resized_image.shape[:2]

                # Center the resized image on a black canvas if needed
                if canvas_width != VIDEO_WIDTH or canvas_height < VIDEO_HEIGHT:
                    canvas = numpy.zeros(
                        (VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=numpy.uint8
                    )

                    # Calculate position to center the image
                    paste_x = (VIDEO_WIDTH - canvas_width) // 2
                    paste_y = (VIDEO_HEIGHT - canvas_height) // 2

                    # Ensure we don't exceed canvas bounds
                    end_y = min(paste_y + canvas_height, VIDEO_HEIGHT)
                    end_x = min(paste_x + canvas_width, VIDEO_WIDTH)

                    canvas[paste_y:end_y, paste_x:end_x] = resized_image[
                        : end_y - paste_y, : end_x - paste_x
                    ]
                    final_image = canvas
                else:
                    final_image = resized_image

                # Create image clip from the optimized image
                image_clip = ImageClip(final_image)

                caption_clips = []
                current_time = 0

                # Create a list to store all audio clips for this slide
                audio_clips = []
                # Store temp file paths to clean up later
                temp_audio_files = []

                logger.info(
                    f"Processing {len(slide.captions)} captions for slide {global_slide_idx}..."
                )

                # Collect all caption data for batch processing
                caption_data = []
                if ENABLE_DUBBING:
                    caption_audio_pairs = zip(slide.captions, slide.caption_dubs)
                else:
                    caption_audio_pairs = [
                        (caption, None) for caption in slide.captions
                    ]

                # First pass: collect caption data and process audio
                for idx, (caption, audio_bytes) in enumerate(caption_audio_pairs):
                    logger.info(
                        f"Processing caption {idx + 1}/{len(slide.captions)}: ENABLE_DUBBING={ENABLE_DUBBING}, has_audio_bytes={audio_bytes is not None}, audio_bytes_size={len(audio_bytes) if audio_bytes else 0}..."
                    )

                    if ENABLE_DUBBING and audio_bytes:
                        # Create temporary audio file
                        with tempfile.NamedTemporaryFile(
                            suffix=".mp3", delete=False
                        ) as temp_audio:
                            temp_audio.write(audio_bytes)
                            temp_audio_path = temp_audio.name
                            temp_audio_files.append(
                                temp_audio_path
                            )  # Track for cleanup
                            logger.info(
                                f"Created temporary audio file at {temp_audio_path} with {len(audio_bytes)} bytes"
                            )

                        try:
                            # Load audio clip to get duration
                            audio_clip = AudioFileClip(temp_audio_path)
                            logger.info(
                                f"Successfully created AudioFileClip from {temp_audio_path}"
                            )

                            # Check if audio clip is valid
                            if audio_clip.duration <= 0:
                                logger.error(
                                    f"Invalid audio clip duration: {audio_clip.duration}"
                                )
                                # Use text-based duration as fallback instead of skipping
                                duration = VideoGenerator._calculate_caption_duration(
                                    caption, DUBBING_WPM
                                )
                                audio_clip = None
                                logger.warning(
                                    f"Using fallback duration: {duration} seconds"
                                )
                            else:
                                duration = audio_clip.duration
                                logger.info(
                                    f"Audio clip duration for caption {idx + 1}: {duration} seconds"
                                )
                        except Exception as e:
                            logger.error(
                                f"Error processing audio clip: {str(e)}", exc_info=True
                            )
                            # Use text-based duration as fallback instead of skipping
                            duration = VideoGenerator._calculate_caption_duration(
                                caption, DUBBING_WPM
                            )
                            audio_clip = None
                            logger.warning(
                                f"Audio processing failed, using fallback duration: {duration} seconds"
                            )
                    else:
                        # Calculate duration based on text length when dubbing is disabled
                        duration = VideoGenerator._calculate_caption_duration(
                            caption, DUBBING_WPM
                        )
                        logger.info(
                            f"Calculated duration for caption {idx + 1}: {duration} seconds (dubbing disabled or no audio_bytes)"
                        )
                        audio_clip = None

                    # Store caption data for batch processing
                    cleaned_caption = caption.replace("\n", " ").strip()
                    caption_data.append(
                        {
                            "text": cleaned_caption,
                            "duration": duration,
                            "audio_clip": audio_clip,
                            "index": idx,
                        }
                    )

                # Batch create text clips for this slide
                if caption_data:
                    logger.info(
                        f"Creating {len(caption_data) * 2} text clips in batch for slide {global_slide_idx}..."
                    )

                    # Prepare text-color pairs for batch creation (main text + shadow for each caption)
                    texts_and_colors = []
                    for data in caption_data:
                        texts_and_colors.extend(
                            [
                                (data["text"], FONT_COLOR),  # Main text
                                (data["text"], "black"),  # Shadow text
                            ]
                        )

                    # Create all text clips in batch
                    batch_text_clips = (
                        await VideoGenerator._create_wrapped_text_clips_batch(
                            texts_and_colors=texts_and_colors,
                            font_size=FONT_SIZE,
                            font_path=FONT_NAME,
                            max_width=VIDEO_WIDTH - 2 * CAPTION_PADDING,
                        )
                    )

                    logger.info(
                        f"Successfully created {len(batch_text_clips)} text clips in batch"
                    )

                # Second pass: apply positioning, timing, and effects
                for idx, data in enumerate(caption_data):
                    duration = data["duration"]
                    audio_clip = data["audio_clip"]

                    # Get the corresponding text clips from batch (main and shadow)
                    text_clip = batch_text_clips[idx * 2]  # Main text
                    shadow_clip = batch_text_clips[idx * 2 + 1]  # Shadow text

                    # Get text clip dimensions for positioning
                    text_width, text_height = text_clip.size

                    # Calculate positioning
                    text_y_pos = max(
                        VIDEO_HEIGHT
                        - text_height
                        - CAPTION_PADDING,  # Simple bottom padding
                        CAPTION_PADDING,  # Ensure text doesn't go above top padding either
                    )

                    # Calculate gradient height to cover the text area
                    gradient_height = min(
                        VIDEO_HEIGHT
                        - text_y_pos
                        + 10,  # From text position to bottom + small buffer
                        VIDEO_HEIGHT
                        // 2,  # Don't let gradient take more than half the screen
                    )

                    # Create gradient background
                    gradient = VideoGenerator._get_gradient_background(
                        width=VIDEO_WIDTH,
                        height=gradient_height,
                        alpha_bottom=1.0,  # Fully opaque at bottom
                        alpha_top=0.0,  # Fully transparent at top
                    )

                    # Create gradient clip
                    gradient_clip = ImageClip(gradient)

                    # Position clips with explicit coordinates and proper padding
                    gradient_y_pos = max(
                        text_y_pos - 10, 0
                    )  # Start gradient slightly above text, but not above frame
                    gradient_clip = gradient_clip.with_position(
                        (0, gradient_y_pos)
                    )  # Full width, positioned from left

                    # Use explicit positioning with proper padding
                    text_x_pos = CAPTION_PADDING  # Explicit left padding
                    shadow_clip = shadow_clip.with_position(
                        (text_x_pos, text_y_pos + 2)
                    ).with_opacity(0.8)
                    text_clip = text_clip.with_position((text_x_pos, text_y_pos))

                    # Set timing for all clips using calculated duration
                    gradient_clip = gradient_clip.with_duration(duration)
                    shadow_clip = shadow_clip.with_duration(duration)
                    text_clip = text_clip.with_duration(duration)

                    # Add fade transitions
                    if (
                        duration > 2 * FADE_DURATION
                    ):  # Only add fades if clip is long enough
                        gradient_clip = gradient_clip.with_effects(
                            [FadeIn(FADE_DURATION), FadeOut(FADE_DURATION)]
                        )
                        shadow_clip = shadow_clip.with_effects(
                            [FadeIn(FADE_DURATION), FadeOut(FADE_DURATION)]
                        )
                        text_clip = text_clip.with_effects(
                            [FadeIn(FADE_DURATION), FadeOut(FADE_DURATION)]
                        )

                    # Set start times
                    gradient_clip = gradient_clip.with_start(current_time)
                    shadow_clip = shadow_clip.with_start(current_time)
                    text_clip = text_clip.with_start(current_time)

                    # Set the start time for audio clip if dubbing is enabled
                    if ENABLE_DUBBING and audio_clip:
                        audio_clip = audio_clip.with_start(current_time)
                        audio_clips.append(audio_clip)
                        logger.info(
                            f"Added audio clip to list for caption {data['index'] + 1}, total audio_clips: {len(audio_clips)}"
                        )
                    else:
                        logger.info(
                            f"No audio clip added for caption {data['index'] + 1}: ENABLE_DUBBING={ENABLE_DUBBING}, audio_clip is None={audio_clip is None}"
                        )

                    current_time += duration

                    # Add clips in order: gradient (bottom), shadow (middle), text (top)
                    caption_clips.extend([gradient_clip, shadow_clip, text_clip])

                # Create composite of image and all captions
                slide_duration = current_time
                image_clip = image_clip.with_duration(slide_duration)

                # Apply zoom effect if enabled
                if ENABLE_ZOOM and ZOOM_FACTOR != 1.0 and slide_duration > 0:
                    logger.info(
                        f"Applying zoom effect to slide {global_slide_idx} - Factor: {ZOOM_FACTOR}, Speed: {ZOOM_STYLE}, Duration: {slide_duration} s"
                    )
                    image_clip = VideoGenerator._apply_zoom_to_clip(
                        image_clip,
                        ZOOM_FACTOR,
                        slide_duration,
                        ZOOM_STYLE,
                        VIDEO_WIDTH,
                        VIDEO_HEIGHT,
                    )
                    logger.info(f"Zoom effect applied to slide {global_slide_idx}")

                # Combine all audio clips for this slide
                if audio_clips:
                    logger.info(
                        f"Adding {len(audio_clips)} audio clips to slide {global_slide_idx}"
                    )
                    slide_audio = CompositeAudioClip(audio_clips)
                    slide_composite = CompositeVideoClip([image_clip] + caption_clips)
                    slide_composite = slide_composite.with_audio(slide_audio)
                    logger.info(f"Successfully added audio to slide {global_slide_idx}")
                else:
                    if ENABLE_DUBBING:
                        logger.warning(
                            f"No audio clips to add to slide {global_slide_idx}"
                        )
                    else:
                        logger.info(
                            f"No audio added to slide {global_slide_idx} (dubbing disabled)"
                        )
                    slide_composite = CompositeVideoClip([image_clip] + caption_clips)

                # Ensure slide_composite has the correct duration
                slide_composite = slide_composite.with_duration(slide_duration)

                # Add fade transitions between slides
                if slide_duration > 2 * FADE_DURATION:
                    slide_composite = slide_composite.with_duration(
                        slide_duration
                    )  # Ensure duration is set
                    slide_composite = slide_composite.with_effects(
                        [FadeIn(FADE_DURATION), FadeOut(FADE_DURATION)]
                    )

                batch_clips.append(slide_composite)

                # Clean up temporary audio files for this slide
                for temp_file in temp_audio_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                            logger.info(f"Removed temporary audio file {temp_file}")
                    except Exception as e:
                        logger.warning(
                            f"Error removing temporary audio file {temp_file}: {e}"
                        )

            # Add batch clips to main list and clean up batch
            slide_clips.extend(batch_clips)

        # Concatenate all slides
        logger.info(f"Concatenating {len(slide_clips)} slide clips...")
        final_clip = concatenate_videoclips(slide_clips, method="compose")
        total_duration = final_clip.duration
        logger.info(f"Final video duration: {total_duration} seconds")

        # Check if final clip has audio with more detailed logging
        if final_clip.audio is None:
            if ENABLE_DUBBING:
                logger.error("Final video has no audio track - debugging slide clips:")
                for i, clip in enumerate(slide_clips):
                    has_audio = clip.audio is not None
                    duration = getattr(clip, "duration", "unknown")
                    logger.error(
                        f"  Slide {i + 1}: has_audio={has_audio}, duration={duration}"
                    )
            else:
                logger.info("Final video has no audio track — dubbing disabled")
        else:
            audio_duration = (
                final_clip.audio.duration
                if hasattr(final_clip.audio, "duration")
                else "unknown"
            )
            logger.info(
                f"Final video has audio track with duration: {audio_duration} seconds"
            )

        # Create a temporary file for video rendering
        temp_output_path = os.path.join(
            tempfile.gettempdir(), f"video_{uuid.uuid4().hex[:8]}.mp4"
        )
        logger.info(f"Writing final video to {temp_output_path}")

        try:
            # Use optimized video writing
            VideoGenerator._save_clip(
                final_clip, temp_output_path, VIDEO_FPS, VIDEO_CODEC, AUDIO_CODEC
            )
            logger.info(f"Successfully wrote video to {temp_output_path}")

            # Read the temporary file into bytes
            with open(temp_output_path, "rb") as f:
                video_bytes = f.read()
            logger.info(f"Read {len(video_bytes)} bytes from {temp_output_path}")

        finally:
            # Clean up temporary file
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
                logger.info(f"Removed temporary file {temp_output_path}")

        # Clean up final clip to free memory
        try:
            final_clip.close()
            logger.info("Cleaned up final clip to free memory")
        except Exception as e:
            logger.warning(f"Error during final cleanup: {e}")

        return video_bytes


if __name__ == "__main__":
    # This is the entry point when running as a Cloud Run Job
    if len(sys.argv) < 2:
        logger.error("Usage: python main.py <creation_id> [cost_centre_id]")
        sys.exit(1)

    creation_id = sys.argv[1]
    cost_centre_id = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None

    async def main():
        try:
            # Load slides from GCS
            gcs_path = f"gs://{GCLOUD_STB_CREATIONS_NAME}/{creation_id}/assets"
            slides = VideoGenerator.load_assets_from_directory(gcs_path)

            if not slides:
                raise ValueError(f"No valid slides found for creation {creation_id}")

            # Generate video with cost_centre_id for tracking TTS costs
            video_bytes = await VideoGenerator.create_video_from_slides(
                slides=slides, cost_centre_id=cost_centre_id
            )

            if ENV == "d":
                # Save video locally in output directory
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                local_video_path = os.path.join(OUTPUT_DIR, f"video_{creation_id}.mp4")
                with open(local_video_path, "wb") as f:
                    f.write(video_bytes)
                logger.info(f"Video saved locally at: {local_video_path}")

            if ENV == "p":
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

    # Run the async main function
    asyncio.run(main())
