import logging
import uuid
from datetime import datetime
from traceback import format_exc

import stripe
from fastapi import HTTPException

from app.models.payment import (
    CreditTransactionType,
    SubscriptionRecord,
    SubscriptionStatus,
    UserSubscriptionResponse,
)
from app.services.firestore import get_firestore_service
from app.services.payments.credit_manager import CreditManager
from app.services.payments.stripe import get_credits_for_subscription_purchase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubscriptionManager:
    """Manages user subscription operations and monthly credit allocation."""

    def __init__(self):
        self.firestore_service = get_firestore_service()
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

            # Create subscription data for Firestore
            subscription_data = {
                "subscription_id": subscription_id,
                "user_id": user_id,
                "stripe_subscription_id": stripe_subscription_id,
                "plan_name": plan_name,
                "status": stripe_sub.status,
                "credits_monthly": monthly_credits,
                "current_period_start": datetime.fromtimestamp(
                    stripe_sub.current_period_start
                ),
                "current_period_end": datetime.fromtimestamp(
                    stripe_sub.current_period_end
                ),
                "cancel_at_period_end": stripe_sub.cancel_at_period_end,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }

            # Insert subscription record into Firestore
            await self.firestore_service.create_document(
                collection_name="users_subscriptions",
                document_data=subscription_data,
                document_id=subscription_id,
            )

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
            # Query user subscriptions from Firestore with automatic conversion
            # Pass the API model class directly - automatic conversion!
            subscriptions = await self.firestore_service.query_collection(
                collection_name="users_subscriptions",
                filters=[("user_id", "==", user_id)],
                order_by="created_at",
                model_class=SubscriptionRecord,  # API model class - automatic conversion!
            )

            # Find active subscription
            active_subscription = None
            for subscription in subscriptions:
                if (
                    subscription.status == SubscriptionStatus.ACTIVE
                    and not active_subscription
                ):
                    active_subscription = subscription

            # Sort by created_at descending (most recent first)
            subscriptions.sort(key=lambda x: x.created_at, reverse=True)

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

            # Find the subscription document by stripe_subscription_id using automatic conversion
            subscriptions = await self.firestore_service.query_collection(
                collection_name="users_subscriptions",
                filters=[("stripe_subscription_id", "==", stripe_subscription_id)],
                limit=1,
                model_class=SubscriptionRecord,  # API model class - automatic conversion!
            )

            if not subscriptions:
                logger.warning(
                    f"No subscription found with stripe_subscription_id: {stripe_subscription_id}"
                )
                return False

            subscription = subscriptions[0]

            # Update subscription data
            update_data = {
                "status": status.value,
                "current_period_start": datetime.fromtimestamp(
                    stripe_sub.current_period_start
                ),
                "current_period_end": datetime.fromtimestamp(
                    stripe_sub.current_period_end
                ),
                "cancel_at_period_end": stripe_sub.cancel_at_period_end,
                "updated_at": datetime.utcnow(),
            }

            # Update the subscription document in Firestore
            await self.firestore_service.update_document(
                collection_name="users_subscriptions",
                document_id=subscription.subscription_id,
                update_data=update_data,
            )

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
            # Get active subscription from Firestore using automatic conversion
            subscriptions = await self.firestore_service.query_collection(
                collection_name="users_subscriptions",
                filters=[
                    ("stripe_subscription_id", "==", stripe_subscription_id),
                    ("status", "==", "active"),
                ],
                limit=1,
                model_class=SubscriptionRecord,  # API model class - automatic conversion!
            )

            if not subscriptions:
                logger.warning(
                    f"No active subscription found for {stripe_subscription_id}"
                )
                return False

            subscription = subscriptions[0]

            # Grant monthly credits
            await self.credit_manager.add_credits(
                subscription.user_id,
                subscription.credits_monthly,
                CreditTransactionType.EARNED_SUBSCRIPTION,
                f"Monthly subscription credits for {subscription.plan_name}",
                subscription.subscription_id,
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
            # Get subscription from Firestore using automatic conversion
            subscription = await self.firestore_service.get_document(
                collection_name="users_subscriptions",
                document_id=subscription_id,
                model_class=SubscriptionRecord,  # API model class - automatic conversion!
            )

            if not subscription or subscription.user_id != user_id:
                raise HTTPException(status_code=404, detail="Subscription not found")

            # Cancel subscription in Stripe (at period end)
            stripe.Subscription.modify(
                subscription.stripe_subscription_id, cancel_at_period_end=True
            )

            # Update local record in Firestore
            update_data = {
                "cancel_at_period_end": True,
                "updated_at": datetime.utcnow(),
            }

            await self.firestore_service.update_document(
                collection_name="users_subscriptions",
                document_id=subscription_id,
                update_data=update_data,
            )

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
