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
    PaymentHistoryResponse,
    PaymentRecord,
    PaymentStatus,
    PaymentType,
    ProductInfo,
)
from app.server.routers.auth_routes import User, get_current_user
from config import STRIPE_SECRET_KEY, STRIPE_WEBHOOK_SECRET

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Stripe
stripe.api_key = STRIPE_SECRET_KEY

# Create payment router
payment_router = APIRouter()

# Initialize BigQuery client
bq_client = bigquery.Client()


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


def determine_payment_type(product: stripe.Product) -> PaymentType:
    """
    Determine payment type from Stripe product metadata.

    Args:
        product: Stripe product object

    Returns:
        PaymentType: The determined payment type
    """
    # Check product metadata for type hints
    metadata = product.metadata or {}

    if "payment_type" in metadata:
        try:
            return PaymentType(metadata["payment_type"])
        except ValueError:
            pass

    # Fallback logic based on product name/description
    name_lower = product.name.lower()
    description_lower = (product.description or "").lower()

    if "credit" in name_lower or "credit" in description_lower:
        return PaymentType.CREDIT_PURCHASE
    elif "premium" in name_lower or "feature" in name_lower:
        return PaymentType.FEATURE_UNLOCK
    else:
        return PaymentType.ONE_TIME


async def fetch_products_from_stripe() -> List[ProductInfo]:
    """
    Fetch products directly from Stripe API.

    Returns:
        List[ProductInfo]: List of available products
    """
    try:
        logger.info("Fetching products from Stripe...")

        # Fetch active products from Stripe
        stripe_products = stripe.Product.list(active=True, limit=100)
        products = []

        for product in stripe_products.data:
            # Get prices for this product
            prices = stripe.Price.list(product=product.id, active=True)

            for price in prices.data:
                # Only include one-time prices (not recurring subscriptions)
                if price.type == "one_time":
                    # Determine payment type from product metadata
                    payment_type = determine_payment_type(product)

                    products.append(
                        ProductInfo(
                            product_id=price.id,  # Use price ID as product ID for payments
                            name=product.name,
                            description=product.description or "",
                            price=price.unit_amount,
                            currency=price.currency,
                            type=payment_type,
                        )
                    )

        logger.info(f"Retrieved {len(products)} products from Stripe")
        return products

    except stripe.error.StripeError as e:
        logger.error(f"Stripe API error: {str(e)}\n{format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to fetch products from Stripe",
        )
    except Exception as e:
        logger.error(f"Failed to fetch products from Stripe: {str(e)}\n{format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve products",
        )


@payment_router.get("/products", response_model=ProductsResponse)
async def get_products() -> ProductsResponse:
    """Get available products for purchase directly from Stripe."""
    products = await fetch_products_from_stripe()
    return ProductsResponse(data=products)


@payment_router.post("/create-checkout-session", response_model=CheckoutSessionResponse)
async def create_checkout_session(
    request: CheckoutSessionRequest,
    current_user: Annotated[User, Depends(get_current_user)],
) -> CheckoutSessionResponse:
    """Create a Stripe Checkout session for web payments."""
    try:
        # Get all available products from Stripe
        products = await fetch_products_from_stripe()

        # Validate product exists
        product = next(
            (p for p in products if p.product_id == request.product_id), None
        )
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        # Create checkout session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=[
                {
                    "price": request.product_id,  # This is the Stripe price ID
                    "quantity": request.quantity,
                }
            ],
            mode="payment",
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

        # Store payment record in database
        payment_id = str(uuid.uuid4())
        await store_payment_record(
            payment_id=payment_id,
            user_id=current_user.username,
            stripe_payment_intent_id=checkout_session.id,
            amount=product.price * request.quantity,
            currency=product.currency,
            status=PaymentStatus.PENDING,
            payment_type=product.type,
            product_id=request.product_id,
            quantity=request.quantity,
            description=product.description,
        )

        logger.info(
            f"Created checkout session {checkout_session.id} for user {current_user.username}"
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


@payment_router.post("/webhook")
async def handle_stripe_webhook(
    request: Request,
    stripe_signature: Annotated[str, Header(alias="stripe-signature")],
):
    """Handle Stripe webhook events."""
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

        # Handle the event
        if event["type"] == "checkout.session.completed":
            checkout_session = event["data"]["object"]
            await handle_checkout_success(checkout_session)
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
                    payment_type=PaymentType(row.payment_type),
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
    payment_type: PaymentType,
    product_id: str,
    quantity: int,
    description: str = None,
) -> None:
    """Store payment record in database."""
    query = """
    INSERT INTO `bai-buchai-p.payments.records` (
        payment_id, user_id, stripe_payment_intent_id, amount, currency,
        status, payment_type, product_id, quantity, description,
        created_at, updated_at
    )
    VALUES (
        @payment_id, @user_id, @stripe_payment_intent_id, @amount, @currency,
        @status, @payment_type, @product_id, @quantity, @description,
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
            bigquery.ScalarQueryParameter("payment_type", "STRING", payment_type.value),
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


async def handle_checkout_success(checkout_session: dict) -> None:
    """Handle successful checkout session."""
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
            f"Checkout successful for user {user_id}, product {product_id}, quantity {quantity}, product type {product_type}"
        )

        # TODO: Implement product-specific logic
        # For example:
        # - Add credits to user account
        # - Unlock premium features
        # - Grant access to specific content

    except Exception as e:
        logger.error(f"Failed to handle checkout success: {str(e)}\n{format_exc()}")
