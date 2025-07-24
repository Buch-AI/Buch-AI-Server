"""
Models Package

This package contains database schema models organized by domain:
- creations.py: Creation and cost centre models
- credits.py: Credit transaction models
- payments.py: Payment record models
- tasks.py: Background task models
- users.py: User, authentication, profile, subscription, and credit models
- shared.py: Common base models, API responses, and embedded models
"""

# Import all models for easy access
from app.models.creations import (
    BaseCostCentre,
    BaseCreationProfile,
    CostCentre,
    CreationProfile,
)
from app.models.credits import (
    BaseCreditTransaction,
    CreditTransaction,
    CreditTransactionType,
)
from app.models.payments import (
    BasePaymentRecord,
    PaymentRecord,
    PaymentStatus,
)
from app.models.shared import (
    CreditBalance,
    CreditBalanceResponse,
    FirestoreBaseModel,
    PaymentHistoryResponse,
    ProductInfo,
    ProductType,
    SocialLink,
    SubscriptionRecord,
    SubscriptionStatus,
    UserSubscriptionResponse,
)
from app.models.tasks import BaseVideoGeneratorTask, VideoGeneratorTask
from app.models.users import (
    BaseUserCredits,
    BaseUserSubscription,
    UserAuth,
    UserCredits,
    UserGeolocation,
    UserProfile,
    UserSubscription,
)

# Collection model mappings for Firestore operations
COLLECTION_MODELS = {
    "creations_cost_centres": CostCentre,
    "creations_profiles": CreationProfile,
    "tasks_video_generator": VideoGeneratorTask,
    "users_auth": UserAuth,
    "users_profiles": UserProfile,
    "users_geolocation": UserGeolocation,
    "users_credits": UserCredits,
    "users_subscriptions": UserSubscription,
    "payments_records": PaymentRecord,
    "credits_transactions": CreditTransaction,
}

__all__ = [
    # Base models
    "FirestoreBaseModel",
    # Creation models
    "BaseCreationProfile",
    "BaseCostCentre",
    "CreationProfile",
    "CostCentre",
    # Credit models
    "BaseCreditTransaction",
    "CreditTransaction",
    "CreditTransactionType",
    # User credit models (from users.py)
    "BaseUserCredits",
    "UserCredits",
    # Payment models
    "BasePaymentRecord",
    "PaymentRecord",
    "PaymentStatus",
    # Task models
    "BaseVideoGeneratorTask",
    "VideoGeneratorTask",
    # User models
    "BaseUserSubscription",
    "UserAuth",
    "UserProfile",
    "UserGeolocation",
    "UserSubscription",
    # Shared API Response models
    "CreditBalance",
    "CreditBalanceResponse",
    "PaymentHistoryResponse",
    "ProductInfo",
    "ProductType",
    "SocialLink",
    "SubscriptionRecord",
    "SubscriptionStatus",
    "UserSubscriptionResponse",
    # Collection mappings
    "COLLECTION_MODELS",
]
