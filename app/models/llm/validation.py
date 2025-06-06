import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Validator:
    """Validator class for LLM-generated content validation."""

    @staticmethod
    def validate_story_split(story_parts: List[List[str]]) -> bool:
        """
        Validates that a story split contains between the minimum and maximum number of parts and subparts.

        Args:
            story_parts: List of parts, where each part is a list of subpart strings

        Returns:
            bool: True if validation passes, False otherwise
        """
        if not isinstance(story_parts, list):
            logger.error("Input must be a list")
            return False

        if not story_parts:
            logger.error("Input cannot be empty")
            return False

        # Check if there are between 3 and 4 parts
        num_parts = len(story_parts)
        if num_parts < 3:
            logger.error(f"Found {num_parts} parts, minimum required is 3")
            return False

        if num_parts > 4:
            logger.error(f"Found {num_parts} parts, maximum allowed is 4")
            return False

        # TODO: Uncomment this once we have a proper validation system.
        # # Check each part to ensure it has between 2 and 3 subparts
        # for i, part in enumerate(story_parts):
        #     if not isinstance(part, list):
        #         logger.error(f"Part {i + 1} must be a list, got {type(part)}")
        #         return False

        #     num_subparts = len(part)

        #     if num_subparts < 2:
        #         logger.error(f"Found {num_subparts} subparts in part {i + 1}, minimum required is 2")
        #         return False

        #     if num_subparts > 3:
        #         logger.error(f"Found {num_subparts} subparts in part {i + 1}, maximum allowed is 3")
        #         return False

        #     # Validate that each subpart is a string
        #     for j, subpart in enumerate(part):
        #         if not isinstance(subpart, str):
        #             logger.error(f"Subpart {j + 1} in part {i + 1} must be a string, got {type(subpart)}")
        #             return False

        logger.info("Validation passed")
        return True
