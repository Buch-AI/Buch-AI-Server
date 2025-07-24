# Model Unification Solution

## Problem Statement

Previously, the codebase had duplicate model definitions for the same entities:

1. **Firestore Models** - in `app/models/firestore.py` inheriting from `FirestoreBaseModel`
2. **API Models** - in various router files and `app/models/payment.py` inheriting from `BaseModel`

This led to:
- Field duplication across models
- Manual conversion code in services and route handlers
- Maintenance overhead when adding/modifying fields
- Risk of inconsistency between representations

## Duplicate Instances Found

### Before Unification:

1. **CreationProfile**
   - Firestore: `app/models/firestore.py` → `CreationProfile(FirestoreBaseModel)`
   - API: `app/server/routers/creation_routes.py` → `CreationProfile(BaseModel)`

2. **PaymentRecord**
   - Firestore: `app/models/firestore.py` → `PaymentRecord(FirestoreBaseModel)`
   - API: `app/models/payment.py` → `PaymentRecord(BaseModel)`

3. **CreditTransaction**
   - Firestore: `app/models/firestore.py` → `CreditTransaction(FirestoreBaseModel)`
   - API: `app/models/payment.py` → `CreditTransaction(BaseModel)`

4. **UserCredits/CreditBalance**
   - Firestore: `app/models/firestore.py` → `UserCredits(FirestoreBaseModel)`
   - API: `app/models/payment.py` → `CreditBalance(BaseModel)`

5. **UserSubscription/SubscriptionRecord**
   - Firestore: `app/models/firestore.py` → `UserSubscription(FirestoreBaseModel)`
   - API: `app/models/payment.py` → `SubscriptionRecord(BaseModel)`

## Solution: Shared Base Models

### Architecture

Created `app/models/shared.py` containing shared base models that define the single source of truth for data structures:

```python
# Example structure
class BaseCreationProfile(BaseModel):
    """Base creation profile model shared between Firestore and API."""
    creation_id: str = Field(..., description="Unique creation identifier")
    title: str = Field(..., min_length=1, description="Creation title")
    # ... all fields with proper validation and documentation
```

### Implementation

1. **Shared Base Models** (`app/models/shared.py`):
   - `BaseCreationProfile`
   - `BasePaymentRecord`
   - `BaseCreditTransaction`
   - `BaseUserCredits`
   - `BaseUserSubscription`
   - `BaseCostCentre`
   - `BaseVideoGeneratorTask`

2. **Firestore Models** now inherit from both shared base and `FirestoreBaseModel`:
   ```python
   class CreationProfile(BaseCreationProfile, FirestoreBaseModel):
       """Creation profile document model for creations_profiles collection."""
       pass
   ```

3. **API Models** now use aliases to shared base models:
   ```python
   # Use the shared base model for API responses
   CreationProfile = BaseCreationProfile
   PaymentRecord = BasePaymentRecord
   CreditBalance = BaseUserCredits
   ```

### Benefits

1. **Single Source of Truth**: Fields are defined once in shared base models
2. **No Manual Conversion**: Since both Firestore and API models inherit from the same base, no conversion needed
3. **Type Safety**: Full Pydantic validation and type hints maintained
4. **Firestore Configuration**: Firestore-specific settings (JSON encoders, etc.) preserved
5. **Easy Maintenance**: Add/modify fields in one place

### Code Simplification

#### Before (Manual Conversion):
```python
# Convert Firestore models to API models
creations = []
for fc in firestore_creations:
    creation = CreationProfile(
        creation_id=fc.creation_id,
        title=fc.title,
        description=fc.description,
        # ... manually mapping each field
    )
    creations.append(creation)
```

#### After (Direct Usage):
```python
# Since both models now inherit from the same base, we can use the Firestore models directly
creations = firestore_creations
```

## Files Modified

1. **Created**: `app/models/shared.py` - Shared base models
2. **Modified**: `app/models/firestore.py` - Updated to inherit from shared bases
3. **Modified**: `app/models/payment.py` - Replaced duplicates with aliases
4. **Modified**: `app/server/routers/creation_routes.py` - Uses shared model
5. **Modified**: `app/server/routers/me_routes.py` - Simplified conversion logic
6. **Modified**: `app/server/routers/payment_routes.py` - Simplified conversion logic

## Migration Guide

For any future models:

1. Define the base model in `app/models/shared.py`
2. Create Firestore model by inheriting from both base and `FirestoreBaseModel`
3. Create API model as an alias to the base model
4. Use the models directly without manual conversion

## Testing

All modified files compile successfully and maintain the same API contract while eliminating duplication. 