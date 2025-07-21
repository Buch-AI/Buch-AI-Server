import logging
import uuid
from datetime import datetime
from traceback import format_exc

import stripe
from fastapi import HTTPException
from google.cloud import bigquery

from app.models.payment import (
    CreditTransactionType,
    SubscriptionRecord,
    SubscriptionStatus,
    UserSubscriptionResponse,
)
from app.services.payments.credit_manager import CreditManager
from app.services.payments.stripe import get_credits_for_subscription_purchase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubscriptionManager:
    """Manages user subscription operations and monthly credit allocation."""

    def __init__(self):
        self.bq_client = bigquery.Client()
        self.credit_manager = CreditManager()

    async def create_subscription(
        self, user_id: str, stripe_subscription_id: str, plan_name: str
    ) -> str:
        """
        Create a new subscription record and grant initial credits.

        Args:
            user_id: The user ID
            stripe_subscription_id: Stripe subscription ID
            plan_name: Name of the subscription plan

        Returns:
            The created subscription ID
        """
        try:
            subscription_id = str(uuid.uuid4())

            # Get subscription details from Stripe
            stripe_sub = stripe.Subscription.retrieve(stripe_subscription_id)

            # Determine monthly credits based on plan
            monthly_credits = get_credits_for_subscription_purchase(plan_name)

            # Insert subscription record
            query = """
            INSERT INTO `bai-buchai-p.users.subscriptions`
            (subscription_id, user_id, stripe_subscription_id, plan_name, status,
             credits_monthly, current_period_start, current_period_end, 
             cancel_at_period_end, created_at, updated_at)
            VALUES (@subscription_id, @user_id, @stripe_subscription_id, @plan_name, @status,
                    @credits_monthly, @current_period_start, @current_period_end,
                    @cancel_at_period_end, CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP())
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "subscription_id", "STRING", subscription_id
                    ),
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                    bigquery.ScalarQueryParameter(
                        "stripe_subscription_id", "STRING", stripe_subscription_id
                    ),
                    bigquery.ScalarQueryParameter("plan_name", "STRING", plan_name),
                    bigquery.ScalarQueryParameter(
                        "status", "STRING", stripe_sub.status
                    ),
                    bigquery.ScalarQueryParameter(
                        "credits_monthly", "INT64", monthly_credits
                    ),
                    bigquery.ScalarQueryParameter(
                        "current_period_start",
                        "TIMESTAMP",
                        datetime.fromtimestamp(stripe_sub.current_period_start),
                    ),
                    bigquery.ScalarQueryParameter(
                        "current_period_end",
                        "TIMESTAMP",
                        datetime.fromtimestamp(stripe_sub.current_period_end),
                    ),
                    bigquery.ScalarQueryParameter(
                        "cancel_at_period_end",
                        "BOOLEAN",
                        stripe_sub.cancel_at_period_end,
                    ),
                ]
            )

            query_job = self.bq_client.query(query, job_config=job_config)
            query_job.result()

            # Grant initial subscription credits
            await self.credit_manager.add_credits(
                user_id,
                monthly_credits,
                CreditTransactionType.EARNED_SUBSCRIPTION,
                f"Initial subscription credits for {plan_name}",
                subscription_id,
            )

            logger.info(f"Created subscription {subscription_id} for user {user_id}")
            return subscription_id

        except Exception as e:
            logger.error(
                f"Failed to create subscription for user {user_id}: {str(e)}\n{format_exc()}"
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to create subscription: {str(e)}"
            )

    async def get_subscriptions(self, user_id: str) -> UserSubscriptionResponse:
        """
        Get all subscriptions for a user.

        Args:
            user_id: The user ID

        Returns:
            UserSubscriptionResponse with all subscriptions and active subscription
        """
        try:
            query = """
            SELECT subscription_id, user_id, stripe_subscription_id, plan_name, status,
                   credits_monthly, current_period_start, current_period_end,
                   cancel_at_period_end, created_at, updated_at
            FROM `bai-buchai-p.users.subscriptions`
            WHERE user_id = @user_id
            ORDER BY created_at DESC
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
                ]
            )

            query_job = self.bq_client.query(query, job_config=job_config)
            results = query_job.result()

            subscriptions = []
            active_subscription = None

            for row in results:
                subscription = SubscriptionRecord(
                    subscription_id=row.subscription_id,
                    user_id=row.user_id,
                    stripe_subscription_id=row.stripe_subscription_id,
                    plan_name=row.plan_name,
                    status=SubscriptionStatus(row.status),
                    credits_monthly=row.credits_monthly,
                    current_period_start=row.current_period_start,
                    current_period_end=row.current_period_end,
                    cancel_at_period_end=row.cancel_at_period_end,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                )
                subscriptions.append(subscription)

                if (
                    subscription.status == SubscriptionStatus.ACTIVE
                    and not active_subscription
                ):
                    active_subscription = subscription

            return UserSubscriptionResponse(
                subscriptions=subscriptions,
                active_subscription=active_subscription,
            )

        except Exception as e:
            logger.error(
                f"Failed to get subscriptions for user {user_id}: {str(e)}\n{format_exc()}"
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to retrieve subscriptions: {str(e)}"
            )

    async def update_subscription_status(
        self, stripe_subscription_id: str, status: SubscriptionStatus
    ) -> bool:
        """
        Update subscription status.

        Args:
            stripe_subscription_id: Stripe subscription ID
            status: New subscription status

        Returns:
            True if successful
        """
        try:
            # Get updated subscription data from Stripe if needed
            stripe_sub = stripe.Subscription.retrieve(stripe_subscription_id)

            update_query = """
            UPDATE `bai-buchai-p.users.subscriptions`
            SET 
                status = @status,
                current_period_start = @current_period_start,
                current_period_end = @current_period_end,
                cancel_at_period_end = @cancel_at_period_end,
                updated_at = CURRENT_TIMESTAMP()
            WHERE stripe_subscription_id = @stripe_subscription_id
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("status", "STRING", status.value),
                    bigquery.ScalarQueryParameter(
                        "current_period_start",
                        "TIMESTAMP",
                        datetime.fromtimestamp(stripe_sub.current_period_start),
                    ),
                    bigquery.ScalarQueryParameter(
                        "current_period_end",
                        "TIMESTAMP",
                        datetime.fromtimestamp(stripe_sub.current_period_end),
                    ),
                    bigquery.ScalarQueryParameter(
                        "cancel_at_period_end",
                        "BOOLEAN",
                        stripe_sub.cancel_at_period_end,
                    ),
                    bigquery.ScalarQueryParameter(
                        "stripe_subscription_id", "STRING", stripe_subscription_id
                    ),
                ]
            )

            query_job = self.bq_client.query(update_query, job_config=job_config)
            query_job.result()

            logger.info(
                f"Updated subscription {stripe_subscription_id} status to {status.value}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to update subscription {stripe_subscription_id}: {str(e)}\n{format_exc()}"
            )
            return False

    async def process_subscription_renewal(self, stripe_subscription_id: str) -> bool:
        """
        Process subscription renewal and grant monthly credits.

        Args:
            stripe_subscription_id: Stripe subscription ID

        Returns:
            True if successful
        """
        try:
            # Get subscription from database
            query = """
            SELECT subscription_id, user_id, plan_name, credits_monthly
            FROM `bai-buchai-p.users.subscriptions`
            WHERE stripe_subscription_id = @stripe_subscription_id
            AND status = 'active'
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "stripe_subscription_id", "STRING", stripe_subscription_id
                    )
                ]
            )

            query_job = self.bq_client.query(query, job_config=job_config)
            results = list(query_job.result())

            if not results:
                logger.warning(
                    f"No active subscription found for {stripe_subscription_id}"
                )
                return False

            row = results[0]

            # Grant monthly credits
            await self.credit_manager.add_credits(
                row.user_id,
                row.credits_monthly,
                CreditTransactionType.EARNED_SUBSCRIPTION,
                f"Monthly subscription credits for {row.plan_name}",
                row.subscription_id,
            )

            # Update subscription period
            await self.update_subscription_status(
                stripe_subscription_id, SubscriptionStatus.ACTIVE
            )

            logger.info(f"Processed renewal for subscription {stripe_subscription_id}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to process renewal for {stripe_subscription_id}: {str(e)}\n{format_exc()}"
            )
            return False

    async def cancel_subscription(self, user_id: str, subscription_id: str) -> bool:
        """
        Cancel a user's subscription (mark for cancellation at period end).

        Args:
            user_id: The user ID
            subscription_id: The subscription ID to cancel

        Returns:
            True if successful
        """
        try:
            # Get subscription from database
            query = """
            SELECT stripe_subscription_id
            FROM `bai-buchai-p.users.subscriptions`
            WHERE subscription_id = @subscription_id
            AND user_id = @user_id
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "subscription_id", "STRING", subscription_id
                    ),
                    bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
                ]
            )

            query_job = self.bq_client.query(query, job_config=job_config)
            results = list(query_job.result())

            if not results:
                raise HTTPException(status_code=404, detail="Subscription not found")

            stripe_subscription_id = results[0].stripe_subscription_id

            # Cancel subscription in Stripe (at period end)
            stripe.Subscription.modify(
                stripe_subscription_id, cancel_at_period_end=True
            )

            # Update local record
            update_query = """
            UPDATE `bai-buchai-p.users.subscriptions`
            SET 
                cancel_at_period_end = TRUE,
                updated_at = CURRENT_TIMESTAMP()
            WHERE subscription_id = @subscription_id
            """

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(
                        "subscription_id", "STRING", subscription_id
                    )
                ]
            )

            query_job = self.bq_client.query(update_query, job_config=job_config)
            query_job.result()

            logger.info(f"Cancelled subscription {subscription_id} for user {user_id}")
            return True

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Failed to cancel subscription {subscription_id}: {str(e)}\n{format_exc()}"
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to cancel subscription: {str(e)}"
            )
