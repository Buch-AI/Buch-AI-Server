import logging
import uuid
from datetime import datetime
from traceback import format_exc
from typing import Annotated, List

import stripe
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from google.cloud import bigquery
from pydantic import BaseModel

from app.models.payment import (
    CreditBalanceResponse,
    CreditTransactionType,
    PaymentHistoryResponse,
    PaymentRecord,
    PaymentStatus,
    ProductInfo,
    ProductType,
    UserSubscriptionResponse,
)
from app.server.routers.auth_routes import User, get_current_user
from app.services.payments.credit_manager import CreditManager
from app.services.payments.stripe import (
    fetch_products,
    get_credits_for_bonus_purchase,
    get_subscription_plan_name,
)
from app.services.payments.subscription_manager import SubscriptionManager
from config import STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Stripe
stripe.api_key = STRIPE_SECRET_KEY

# Create payment router
payment_router = APIRouter()

# Initialize BigQuery client and managers
bq_client = bigquery.Client()
credit_manager = CreditManager()
subscription_manager = SubscriptionManager()


class ProductsResponse(BaseModel):
    data: List[ProductInfo]


class CheckoutSessionRequest(BaseModel):
    product_id: str
    quantity: int = 1
    success_url: str
    cancel_url: str


class CheckoutSessionResponse(BaseModel):
    checkout_url: str
    session_id: str


@payment_router.get("/products", response_model=ProductsResponse)
async def get_products() -> ProductsResponse:
    """Get available products for purchase directly from Stripe."""
    products = await fetch_products()
    return ProductsResponse(data=products)


@payment_router.post("/create-checkout-session", response_model=CheckoutSessionResponse)
async def create_checkout_session(
    request: CheckoutSessionRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> CheckoutSessionResponse:
    """Create a Stripe Checkout session for both one-time and subscription payments."""
    try:
        # Get all available products from Stripe
        products = await fetch_products()

        # Validate product exists
        product = next(
            (p for p in products if p.product_id == request.product_id), None
        )
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        # Determine checkout mode based on payment type
        checkout_mode = (
            "subscription" if product.type == ProductType.SUBSCRIPTION else "payment"
        )

        # Create checkout session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price": request.product_id,  # This is the Stripe price ID
                    "quantity": request.quantity
                    if product.type == ProductType.BONUS
                    else 1,
                }
            ],
            mode=checkout_mode,
            success_url=request.success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=request.cancel_url,
            client_reference_id=current_user.username,
            metadata={
                "user_id": current_user.username,
                "product_id": request.product_id,
                "quantity": str(request.quantity),
                "product_type": product.type.value,
            },
        )

        # Store payment record for one-time payments
        if product.type == ProductType.BONUS:
            payment_id = str(uuid.uuid4())
            await store_payment_record(
                payment_id=payment_id,
                user_id=current_user.username,
                stripe_payment_intent_id=checkout_session.id,
                amount=product.price * request.quantity,
                currency=product.currency,
                status=PaymentStatus.PENDING,
                product_type=product.type,
                product_id=request.product_id,
                quantity=request.quantity,
                description=product.description,
            )

        logger.info(
            f"Created {checkout_mode} checkout session {checkout_session.id} for user {current_user.username}"
        )

        return CheckoutSessionResponse(
            checkout_url=checkout_session.url,
            session_id=checkout_session.id,
        )

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}\n{format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Payment processing error: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Failed to create checkout session: {str(e)}\n{format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create checkout session",
        )


@payment_router.get("/credits", response_model=CreditBalanceResponse)
async def get_user_credits(
    current_user: Annotated[User, Depends(get_current_user)],
) -> CreditBalanceResponse:
    """Get user's credit balance."""
    credits = await credit_manager.get_credits(current_user.username)
    if not credits:
        # Initialize credits for new user
        await credit_manager._ensure_user_credit_record(current_user.username)
        credits = await credit_manager.get_credits(current_user.username)

    return CreditBalanceResponse(
        user_id=credits.user_id,
        balance=credits.balance,
        total_earned=credits.total_earned,
        total_spent=credits.total_spent,
        last_updated=credits.last_updated,
    )


@payment_router.get("/credits/transactions")
async def get_credit_transactions(
    current_user: Annotated[User, Depends(get_current_user)],
    limit: int = 50,
    offset: int = 0,
):
    """Get user's credit transaction history."""
    transactions = await credit_manager.get_credit_transactions(
        current_user.username, limit, offset
    )
    return {"transactions": transactions}


@payment_router.get("/subscriptions", response_model=UserSubscriptionResponse)
async def get_user_subscriptions(
    current_user: Annotated[User, Depends(get_current_user)],
) -> UserSubscriptionResponse:
    """Get user's subscription information."""
    return await subscription_manager.get_subscriptions(current_user.username)


@payment_router.post("/subscriptions/{subscription_id}/cancel")
async def cancel_subscription(
    subscription_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Cancel a user's subscription."""
    success = await subscription_manager.cancel_subscription(
        current_user.username, subscription_id
    )
    if success:
        return {"message": "Subscription cancelled successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to cancel subscription")


@payment_router.post("/webhook")
async def handle_stripe_webhook(
    request: Request,
    stripe_signature: Annotated[str, Header(alias="stripe-signature")],
):
    """Handle Stripe webhook events for payments and subscriptions."""
    try:
        # Get request body
        body = await request.body()

        # Verify webhook signature
        try:
            event = stripe.Webhook.construct_event(
                body, stripe_signature, STRIPE_WEBHOOK_SECRET
            )
        except ValueError:
            logger.error("Invalid payload in webhook")
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError:
            logger.error("Invalid signature in webhook")
            raise HTTPException(status_code=400, detail="Invalid signature")

        # Handle different event types
        if event["type"] == "checkout.session.completed":
            checkout_session = event["data"]["object"]
            if checkout_session["mode"] == "subscription":
                await handle_subscription_checkout_success(checkout_session)
            else:
                await handle_bonus_checkout_success(checkout_session)
        elif event["type"] == "invoice.payment_succeeded":
            await handle_invoice_payment_succeeded(event["data"]["object"])
        elif event["type"] == "customer.subscription.updated":
            await handle_subscription_updated(event["data"]["object"])
        elif event["type"] == "customer.subscription.deleted":
            await handle_subscription_canceled(event["data"]["object"])
        else:
            logger.info(f"Unhandled event type: {event['type']}")

        return {"status": "success"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook error: {str(e)}\n{format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook processing failed",
        )


@payment_router.get("/history", response_model=PaymentHistoryResponse)
async def get_payment_history(
    current_user: Annotated[User, Depends(get_current_user)],
    limit: int = 50,
    offset: int = 0,
) -> PaymentHistoryResponse:
    """Get payment history for the current user."""
    try:
        # Query payments from database
        query = """
        SELECT *
        FROM `bai-buchai-p.payments.records`
        WHERE user_id = @user_id
        ORDER BY created_at DESC
        LIMIT @limit OFFSET @offset
        """

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "user_id", "STRING", current_user.username
                ),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
                bigquery.ScalarQueryParameter("offset", "INT64", offset),
            ]
        )

        query_job = bq_client.query(query, job_config=job_config)
        results = query_job.result()

        payments = []
        for row in results:
            payments.append(
                PaymentRecord(
                    payment_id=row.payment_id,
                    user_id=row.user_id,
                    stripe_payment_intent_id=row.stripe_payment_intent_id,
                    amount=row.amount,
                    currency=row.currency,
                    status=PaymentStatus(row.status),
                    product_type=ProductType(row.product_type),
                    product_id=row.product_id,
                    quantity=row.quantity,
                    description=row.description,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    completed_at=row.completed_at,
                )
            )

        # Get total count
        count_query = """
        SELECT COUNT(*) as total
        FROM `bai-buchai-p.payments.records`
        WHERE user_id = @user_id
        """

        count_job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter(
                    "user_id", "STRING", current_user.username
                ),
            ]
        )

        count_job = bq_client.query(count_query, count_job_config)
        total_count = list(count_job.result())[0].total

        return PaymentHistoryResponse(payments=payments, total_count=total_count)

    except Exception as e:
        logger.error(f"Failed to get payment history: {str(e)}\n{format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve payment history",
        )


async def store_payment_record(
    payment_id: str,
    user_id: str,
    stripe_payment_intent_id: str,
    amount: int,
    currency: str,
    status: PaymentStatus,
    product_type: ProductType,
    product_id: str,
    quantity: int,
    description: str = None,
) -> None:
    """Store payment record in database."""
    query = """
    INSERT INTO `bai-buchai-p.payments.records` (
        payment_id, user_id, stripe_payment_intent_id, amount, currency,
        status, product_type, product_id, quantity, description,
        created_at, updated_at
    )
    VALUES (
        @payment_id, @user_id, @stripe_payment_intent_id, @amount, @currency,
        @status, @product_type, @product_id, @quantity, @description,
        CURRENT_TIMESTAMP(), CURRENT_TIMESTAMP()
    )
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("payment_id", "STRING", payment_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            bigquery.ScalarQueryParameter(
                "stripe_payment_intent_id", "STRING", stripe_payment_intent_id
            ),
            bigquery.ScalarQueryParameter("amount", "INT64", amount),
            bigquery.ScalarQueryParameter("currency", "STRING", currency),
            bigquery.ScalarQueryParameter("status", "STRING", status.value),
            bigquery.ScalarQueryParameter("product_type", "STRING", product_type.value),
            bigquery.ScalarQueryParameter("product_id", "STRING", product_id),
            bigquery.ScalarQueryParameter("quantity", "INT64", quantity),
            bigquery.ScalarQueryParameter("description", "STRING", description),
        ]
    )

    query_job = bq_client.query(query, job_config=job_config)
    query_job.result()


async def update_payment_status(
    stripe_payment_intent_id: str, status: PaymentStatus, completed_at: datetime = None
) -> None:
    """Update payment status in database."""
    query = """
    UPDATE `bai-buchai-p.payments.records`
    SET status = @status, updated_at = CURRENT_TIMESTAMP()
    """

    parameters = [
        bigquery.ScalarQueryParameter("status", "STRING", status.value),
        bigquery.ScalarQueryParameter(
            "stripe_payment_intent_id", "STRING", stripe_payment_intent_id
        ),
    ]

    if completed_at:
        query += ", completed_at = @completed_at"
        parameters.append(
            bigquery.ScalarQueryParameter("completed_at", "TIMESTAMP", completed_at)
        )

    query += " WHERE stripe_payment_intent_id = @stripe_payment_intent_id"

    job_config = bigquery.QueryJobConfig(query_parameters=parameters)
    query_job = bq_client.query(query, job_config=job_config)
    query_job.result()


async def handle_bonus_checkout_success(checkout_session: dict) -> None:
    """Handle successful one-time payment checkout session."""
    try:
        # Update payment status using checkout session ID
        await update_payment_status(
            checkout_session["id"], PaymentStatus.COMPLETED, datetime.utcnow()
        )

        # Process the payment based on metadata
        metadata = checkout_session.get("metadata", {})
        user_id = metadata.get("user_id")
        product_id = metadata.get("product_id")
        quantity = int(metadata.get("quantity", 1))
        product_type = metadata.get("product_type")

        logger.info(
            f"One-time payment successful for user {user_id}, product {product_id}, quantity {quantity}"
        )

        # For bonus payments, add credits immediately
        if product_type == ProductType.BONUS.value:
            # Calculate credits based on Stripe metadata
            credit_amount = get_credits_for_bonus_purchase(product_id, quantity)
            await credit_manager.add_credits(
                user_id,
                credit_amount,
                CreditTransactionType.EARNED_BONUS,
                "Credits from bonus purchase",
                checkout_session["id"],
            )

    except Exception as e:
        logger.error(f"Failed to handle checkout success: {str(e)}\n{format_exc()}")


async def handle_subscription_checkout_success(checkout_session: dict) -> None:
    """Handle successful subscription checkout session."""
    try:
        subscription_id = checkout_session["subscription"]
        user_id = checkout_session["client_reference_id"]

        # Get plan name from Stripe subscription
        plan_name = get_subscription_plan_name(subscription_id)

        await subscription_manager.create_subscription(
            user_id, subscription_id, plan_name
        )

        logger.info(f"Created subscription for user {user_id}")

    except Exception as e:
        logger.error(
            f"Failed to handle subscription checkout: {str(e)}\n{format_exc()}"
        )


async def handle_invoice_payment_succeeded(invoice: dict) -> None:
    """Handle successful recurring payment (monthly subscription renewal)."""
    try:
        subscription_id = invoice["subscription"]

        # Process subscription renewal and grant monthly credits
        await subscription_manager.process_subscription_renewal(subscription_id)

        logger.info(f"Processed invoice payment for subscription {subscription_id}")

    except Exception as e:
        logger.error(f"Failed to handle invoice payment: {str(e)}\n{format_exc()}")


async def handle_subscription_updated(subscription: dict) -> None:
    """Handle subscription status updates."""
    try:
        from app.models.payment import SubscriptionStatus

        stripe_subscription_id = subscription["id"]
        status = SubscriptionStatus(subscription["status"])

        await subscription_manager.update_subscription_status(
            stripe_subscription_id, status
        )

        logger.info(
            f"Updated subscription {stripe_subscription_id} status to {status.value}"
        )

    except Exception as e:
        logger.error(f"Failed to handle subscription update: {str(e)}\n{format_exc()}")


async def handle_subscription_canceled(subscription: dict) -> None:
    """Handle subscription cancellation."""
    try:
        from app.models.payment import SubscriptionStatus

        stripe_subscription_id = subscription["id"]

        await subscription_manager.update_subscription_status(
            stripe_subscription_id, SubscriptionStatus.CANCELED
        )

        logger.info(f"Cancelled subscription {stripe_subscription_id}")

    except Exception as e:
        logger.error(
            f"Failed to handle subscription cancellation: {str(e)}\n{format_exc()}"
        )
