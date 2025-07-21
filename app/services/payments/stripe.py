import logging
from traceback import format_exc
from typing import List

import stripe
from fastapi import HTTPException, status

from app.models.payment import ProductInfo, ProductType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def fetch_products() -> List[ProductInfo]:
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
                # Determine payment type from product metadata and price type
                product_type = determine_product_type(product, price)

                # Support both one-time and recurring prices
                if price.type == "one_time" or price.type == "recurring":
                    products.append(
                        ProductInfo(
                            product_id=price.id,  # Use price ID as product ID for payments
                            name=product.name,
                            description=product.description or "",
                            price=price.unit_amount,
                            currency=price.currency,
                            type=product_type,
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


def determine_product_type(product: stripe.Product, price: stripe.Price) -> ProductType:
    """
    Determine payment type from Stripe product metadata and price type.

    Args:
        product: Stripe product object
        price: Stripe price object

    Returns:
        ProductType: The determined payment type
    """
    # Check product metadata for explicit type hints first
    metadata = product.metadata or {}

    if "product_type" in metadata:
        try:
            return ProductType(metadata["product_type"])
        except ValueError:
            pass

    # Check price metadata for explicit type hints
    price_metadata = price.metadata or {}
    if "product_type" in price_metadata:
        try:
            return ProductType(price_metadata["product_type"])
        except ValueError:
            pass

    # Use price type as primary indicator - this is the most reliable method
    if price.type == "recurring":
        return ProductType.SUBSCRIPTION
    elif price.type == "one_time":
        return ProductType.BONUS

    # Fallback logic based on product name/description (if price type is somehow not available)
    name_lower = product.name.lower()

    if "subscription" in name_lower or "monthly" in name_lower or "plan" in name_lower:
        return ProductType.SUBSCRIPTION
    else:
        return ProductType.BONUS


def get_subscription_plan_name(stripe_subscription_id: str) -> str:
    """
    Get plan name from Stripe subscription.

    Args:
        stripe_subscription_id: Stripe subscription ID

    Returns:
        Plan name (price nickname or default "basic")
    """
    try:
        stripe_sub = stripe.Subscription.retrieve(stripe_subscription_id)
        plan_name = stripe_sub.items.data[0].price.nickname or "basic"
        return plan_name
    except Exception as e:
        logger.error(
            f"Failed to get plan name for subscription {stripe_subscription_id}: {str(e)}"
        )
        return "basic"  # Default fallback


def get_credits_for_bonus_purchase(product_id: str, quantity: int) -> int:
    """
    Calculate credits for a bonus purchase based on Stripe product metadata.

    Args:
        product_id: Stripe price ID
        quantity: Purchase quantity

    Returns:
        Number of credits to grant
    """
    try:
        # Get price and product information
        price = stripe.Price.retrieve(product_id)

        # Check price metadata first
        if price.metadata and "credits_permanent" in price.metadata:
            try:
                credits_per_item = int(price.metadata["credits_permanent"])
                return credits_per_item * quantity
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid credits_permanent metadata for price {price.id}"
                )

        # Check product metadata
        product = stripe.Product.retrieve(price.product)
        if product.metadata and "credits_permanent" in product.metadata:
            try:
                credits_per_item = int(product.metadata["credits_permanent"])
                return credits_per_item * quantity
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid credits_permanent metadata for product {product.id}"
                )

        # Fallback: 1 credit per dollar spent
        credit_amount = (price.unit_amount * quantity) // 100
        logger.info(
            f"Using fallback credit calculation: {credit_amount} credits for ${price.unit_amount * quantity / 100}"
        )
        return credit_amount

    except Exception as e:
        logger.error(f"Failed to calculate credits for product {product_id}: {str(e)}")
        # TODO: Payment!
        # Fallback calculation if all else fails
        return quantity * 10  # Default 10 credits per item


def get_credits_for_subscription_purchase(plan_name: str) -> int:
    """
    Get monthly credit allocation based on plan name from Stripe.

    Args:
        plan_name: The plan name (should match Stripe price nickname)

    Returns:
        Number of monthly credits for the plan
    """
    try:
        # Search for active prices with matching nickname
        prices = stripe.Price.list(active=True, type="recurring", limit=100)

        for price in prices.data:
            # Check if nickname matches plan name
            if price.nickname and price.nickname.lower() == plan_name.lower():
                # Check for credits_monthly in metadata
                if price.metadata and "credits_monthly" in price.metadata:
                    try:
                        return int(price.metadata["credits_monthly"])
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid credits_monthly metadata for price {price.id}: {price.metadata['credits_monthly']}"
                        )

                # If no metadata, check the product metadata
                product = stripe.Product.retrieve(price.product)
                if product.metadata and "credits_monthly" in product.metadata:
                    try:
                        return int(product.metadata["credits_monthly"])
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid credits_monthly metadata for product {product.id}: {product.metadata['credits_monthly']}"
                        )

        # Fallback: check by plan name in product name or description
        products = stripe.Product.list(active=True, limit=100)
        for product in products.data:
            product_name = product.name.lower()
            if plan_name.lower() in product_name:
                if product.metadata and "credits_monthly" in product.metadata:
                    try:
                        return int(product.metadata["credits_monthly"])
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid credits_monthly metadata for product {product.id}: {product.metadata['credits_monthly']}"
                        )

        logger.warning(
            f"No credits_monthly found for plan '{plan_name}' in Stripe metadata, using default"
        )

    except stripe.error.StripeError as e:
        logger.error(
            f"Stripe error while fetching plan credits for '{plan_name}': {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error fetching plan credits for '{plan_name}': {str(e)}")

    # TODO: Payment!
    # Default fallback values if not found in Stripe
    plan_credits = {
        "basic": 100,
        "premium": 300,
        "pro": 500,
        "starter": 50,
    }

    return plan_credits.get(plan_name.lower(), 100)
