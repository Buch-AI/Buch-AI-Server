import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from google.cloud import texttospeech
from google.cloud.texttospeech_v1.types import (
    AudioConfig,
    SynthesisInput,
    VoiceSelectionParams,
)

from app.models.cost_centre import CostCentreManager
from app.tasks.video_generator.video_slide import VideoSlide

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize cost centre manager
cost_centre_manager = CostCentreManager()


class Dubber(ABC):
    """Abstract base class for audio generation in video creation.

    This class defines the interface for generating audio files that will be used
    in video generation. Different implementations can provide different audio
    generation strategies (e.g., text-to-speech, pre-recorded audio, etc.).
    """

    @abstractmethod
    async def create_audio_from_slides(
        self, slides: List[VideoSlide]
    ) -> List[VideoSlide]:
        """Generate audio content from a list of video slides.

        Args:
            slides: List of VideoSlide objects containing the content to be converted to audio

        Returns:
            List[VideoSlide]: List of VideoSlide objects with the generated audio file as bytes

        Raises:
            NotImplementedError: If the method is not implemented by a concrete class
        """
        raise NotImplementedError("Subclasses must implement create_audio_from_slides")


class GoogleCloudDubber(Dubber):
    """Concrete implementation of Dubber using Google Cloud Text-to-Speech API.

    This class generates audio files from text captions using Google Cloud's
    Text-to-Speech service. It supports multiple languages and voices.
    """

    def __init__(
        self,
        language_code: str = "en-US",
        voice_name: str = "en-US-Neural2-J",
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
        cost_centre_id: Optional[str] = None,
    ):
        """Initialize the Google Cloud Dubber.

        Args:
            language_code: Language code for the voice (e.g., "en-US")
            voice_name: Specific voice name from Google Cloud TTS
            speaking_rate: Speaking rate multiplier (0.25 to 4.0)
            pitch: Voice pitch adjustment (-20.0 to 20.0)
            cost_centre_id: Optional ID for tracking costs of TTS usage
        """
        logger.info(
            f"Initializing GoogleCloudDubber with language_code={language_code}, voice_name={voice_name}"
        )
        self.client = texttospeech.TextToSpeechClient()
        self.language_code = language_code
        self.voice_name = voice_name
        self.speaking_rate = speaking_rate
        self.pitch = pitch
        self.cost_centre_id = cost_centre_id

        # Determine voice type and set cost per character
        if "Neural2" in self.voice_name:
            # Neural2 voices cost $0.000016 per character
            self.cost_per_character = 0.000016
            logger.info("Using Neural2 voice: $0.000016 per character")
        elif "Wavenet" in self.voice_name:
            # WaveNet voices cost $0.000016 per character
            self.cost_per_character = 0.000016
            logger.info("Using WaveNet voice: $0.000016 per character")
        elif "Chirp" in self.voice_name:
            # Chirp HD voices cost $0.00003 per character
            self.cost_per_character = 0.00003
            logger.info("Using Chirp HD voice: $0.00003 per character")
        elif "Studio" in self.voice_name:
            # Studio voices cost $0.00016 per character
            self.cost_per_character = 0.00016
            logger.info("Using Studio voice: $0.00016 per character")
        else:
            # Standard voices cost $0.000004 per character
            self.cost_per_character = 0.000004
            logger.info("Using Standard voice: $0.000004 per character")

    def calculate_cost(self, text: str) -> float:
        """Calculate the cost of synthesizing the given text.

        Args:
            text: The text to synthesize

        Returns:
            float: The cost in USD
        """
        # Count characters including spaces and punctuation
        character_count = len(text)
        cost = character_count * self.cost_per_character
        return cost

    def _synthesize_speech(self, text: str) -> bytes:
        """Synthesize speech from text using Google Cloud TTS.

        Args:
            text: Text to convert to speech

        Returns:
            bytes: Audio content in MP3 format
        """
        logger.info(
            f"Synthesizing speech for text: {text[:50]}..."
        )  # Log first 50 chars

        try:
            synthesis_input = SynthesisInput(text=text)

            voice = VoiceSelectionParams(
                language_code=self.language_code,
                name=self.voice_name,
            )

            audio_config = AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=self.speaking_rate,
                pitch=self.pitch,
            )

            logger.info(
                f"Making TTS request with voice={voice}, audio_config={audio_config}"
            )
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config,
            )

            if not response or not response.audio_content:
                logger.error("Received empty response from Google Cloud TTS")
                raise ValueError("Received empty response from Google Cloud TTS")

            audio_size = len(response.audio_content)
            logger.info(f"Successfully synthesized audio of size {audio_size} bytes")

            # Log cost information for debugging
            if self.cost_centre_id:
                logger.info(f"Calculating cost for {len(text)} characters")
                cost = self.calculate_cost(text)
                logger.info(f"Cost for text: ${cost:.6f}")

            return response.audio_content

        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}", exc_info=True)
            raise

    async def create_audio_from_slides(
        self, slides: List[VideoSlide]
    ) -> List[VideoSlide]:
        """Generate audio content from a list of video slides.

        Args:
            slides: List of VideoSlide objects containing the content to be converted to audio

        Returns:
            List[VideoSlide]: List of VideoSlide objects with the generated audio file as bytes
        """
        logger.info(f"Processing {len(slides)} slides for audio generation")
        processed_slides = []
        total_characters = 0
        total_cost = 0.0

        for idx, slide in enumerate(slides, 1):
            logger.info(
                f"Processing slide {idx}/{len(slides)} with {len(slide.captions)} captions"
            )

            try:
                # Generate audio for each caption
                caption_dubs = []
                for caption_idx, caption in enumerate(slide.captions, 1):
                    logger.info(
                        f"Generating audio for caption {caption_idx}/{len(slide.captions)}"
                    )
                    audio_content = self._synthesize_speech(caption)

                    if not audio_content:
                        logger.error(
                            f"Empty audio content received for caption {caption_idx}"
                        )
                        raise ValueError(
                            f"Empty audio content for caption {caption_idx}"
                        )

                    caption_dubs.append(audio_content)

                    # Track characters and cost
                    total_characters += len(caption)
                    total_cost += self.calculate_cost(caption)

                # Create a new slide with the generated audio
                processed_slide = VideoSlide(
                    image=slide.image,
                    captions=slide.captions,
                    caption_dubs=caption_dubs,
                )
                processed_slides.append(processed_slide)
                logger.info(f"Successfully processed slide {idx}")

            except Exception as e:
                logger.error(f"Error processing slide {idx}: {str(e)}", exc_info=True)
                raise

        logger.info(f"Successfully processed all {len(processed_slides)} slides")

        # Update cost centre with total cost if cost_centre_id is provided
        if self.cost_centre_id and total_cost > 0:
            logger.info(f"Total characters processed: {total_characters}")
            logger.info(f"Total cost for all slides: ${total_cost:.6f}")
            await cost_centre_manager.update_cost_centre(
                self.cost_centre_id, total_cost
            )

        return processed_slides
