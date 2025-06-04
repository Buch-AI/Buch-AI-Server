import logging
import uuid
from traceback import format_exc

from fastapi import HTTPException
from google.cloud import bigquery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CostCentreManager:
    async def create_cost_centre(
        self,
        creation_id: str,
        user_id: str,
    ) -> str:
        """Generate a new cost centre for a specific creation."""
        # Initialize clients
        bigquery_client = bigquery.Client()

        # First, verify the creation belongs to the user
        verify_query = """
      SELECT creation_id
      FROM `bai-buchai-p.creations.profiles`
      WHERE creation_id = @creation_id
      AND user_id = @user_id
      """

        verify_job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("creation_id", "STRING", creation_id),
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            ]
        )

        verify_job = bigquery_client.query(verify_query, verify_job_config)
        if not list(verify_job.result()):
            logger.error(
                f"Creation {creation_id} not found or unauthorized\n{format_exc()}"
            )
            raise HTTPException(
                status_code=404,
                detail="Creation not found or you don't have permission to create a cost centre for it",
            )

        # Generate a unique cost centre ID
        cost_centre_id = str(uuid.uuid4())

        # Insert the cost centre
        query = """
      INSERT INTO `bai-buchai-p.creations.cost_centres` (
          cost_centre_id, creation_id, user_id, created_at, cost
      )
      VALUES (
          @cost_centre_id,
          @creation_id,
          @user_id,
          CURRENT_TIMESTAMP(),
          0
      )
      """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "cost_centre_id", "STRING", cost_centre_id
                ),
                bigquery.ScalarQueryParameter("creation_id", "STRING", creation_id),
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            ]
        )

        query_job = bigquery_client.query(query, job_config=job_config)
        query_job.result()  # Wait for the query to complete

        return cost_centre_id

    async def update_cost_centre(self, cost_centre_id: str, cost: float) -> None:
        """Update the cost for a cost centre."""
        if not cost_centre_id:
            return

        try:
            bigquery_client = bigquery.Client()
            query = """
          UPDATE `bai-buchai-p.creations.cost_centres`
          SET cost = cost + @additional_cost
          WHERE cost_centre_id = @cost_centre_id
          """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("additional_cost", "NUMERIC", cost),
                    bigquery.ScalarQueryParameter(
                        "cost_centre_id", "STRING", cost_centre_id
                    ),
                ]
            )

            # Use query instead of query_async
            query_job = bigquery_client.query(query, job_config=job_config)
            query_job.result()  # Wait for the query to complete
        except Exception as e:
            logging.error(f"Failed to update cost centre: {e}")
            # Don't raise exception to prevent disrupting the main flow

    async def delete_cost_centre(self, cost_centre_id: str, user_id: str) -> bool:
        """Delete a cost centre if it exists and belongs to the user."""
        if not cost_centre_id:
            return False

        try:
            bigquery_client = bigquery.Client()

            # First verify the cost centre belongs to the user
            verify_query = """
          SELECT cost_centre_id
          FROM `bai-buchai-p.creations.cost_centres`
          WHERE cost_centre_id = @cost_centre_id
          AND user_id = @user_id
          """

            verify_job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "cost_centre_id", "STRING", cost_centre_id
                    ),
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                ]
            )

            verify_job = bigquery_client.query(verify_query, verify_job_config)
            if not list(verify_job.result()):
                logger.error(
                    f"Cost centre {cost_centre_id} not found or unauthorized\n{format_exc()}"
                )
                return False

            # Delete the cost centre
            delete_query = """
          DELETE FROM `bai-buchai-p.creations.cost_centres`
          WHERE cost_centre_id = @cost_centre_id
          AND user_id = @user_id
          """

            delete_job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "cost_centre_id", "STRING", cost_centre_id
                    ),
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                ]
            )

            delete_job = bigquery_client.query(delete_query, delete_job_config)
            delete_job.result()  # Wait for the query to complete

            return True
        except Exception as e:
            logger.error(f"Failed to delete cost centre: {e}\n{format_exc()}")
            return False
