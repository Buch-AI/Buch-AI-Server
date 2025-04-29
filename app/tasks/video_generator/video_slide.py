from typing import List, Optional

from pydantic import BaseModel, Field


class VideoSlide(BaseModel):
    """Model representing a single slide in the video."""

    image: bytes = Field(..., description="Raw bytes of the image")
    captions: List[str] = Field(
        ..., min_items=1, description="List of captions to show sequentially"
    )
    caption_dubs: Optional[List[bytes]] = Field(
        default=None,
        description="Optional list of audio files to play sequentially. If not provided, audio will be generated.",
    )
