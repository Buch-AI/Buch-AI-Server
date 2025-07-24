# Credit System Update: Two-Pool Architecture

## Overview

The credit system has been updated to support a two-pool architecture where user credits are split into two distinct pools:

- **`credits_monthly`**: Credits from subscriptions that are topped up monthly
- **`credits_permanent`**: Credits from bonuses, add-ons, or promotional activities that stay with the user permanently

## Key Changes Made

### 1. Data Models Updated

**`app/models/shared.py`:**
- Updated `CreditBalance` model to include `credits_monthly` and `credits_permanent` fields
- Added a computed `balance` property that returns the sum of both pools
- Updated `CreditBalanceResponse` to include both pool fields

**`app/models/users.py`:**
- Updated `BaseUserCredits` model with the same two-pool structure
- Added computed `balance` property

**`app/models/credits.py`:**
- Added `CreditPool` enum with `MONTHLY` and `PERMANENT` values
- Updated `BaseCreditTransaction` to include optional `pool` field

### 2. Credit Manager Logic Updated

**`app/services/payments/credit_manager.py`:**
- Updated `add_credits()` method to:
  - Accept optional `pool` parameter
  - Auto-determine pool based on transaction type if not specified
  - Add credits to the appropriate pool (monthly/permanent)
  
- Updated `spend_credits()` method to:
  - **Prioritize spending from permanent pool first, then monthly pool**
  - Create separate transaction records for each pool spent from
  - Maintain detailed logging of which pools were used

- Updated `_ensure_user_credit_record()` to initialize both credit pools
- Updated `_record_credit_transaction()` to track pool information

### 3. API Updates

**`app/server/routers/payment_routes.py`:**
- Updated credit balance endpoint to return both credit pools
- Updated bonus credit allocation to use `CreditPool.PERMANENT`

### 4. Subscription Management

**`app/services/payments/subscription_manager.py`:**
- Updated subscription credit grants to use `CreditPool.MONTHLY`
- Updated renewal process to properly categorize monthly credits

### 5. Database Schema Updates

**Firestore (`terraform/firestore.rules`):**
- Updated `users_credits` collection rules to require both `credits_monthly` and `credits_permanent` fields
- Updated `credits_transactions` collection rules to allow optional `pool` field

**BigQuery (`terraform/main.tf`):**
- Updated `users_credits` table schema to replace `balance` with `credits_monthly` and `credits_permanent`
- Added `pool` field to `credits_transactions` table schema

## Credit Spending Priority

When a user spends credits, the system follows this priority order:

1. **Permanent Credits First**: Deduct from `credits_permanent` pool
2. **Monthly Credits Second**: Deduct remaining amount from `credits_monthly` pool

This ensures that permanent credits (which are more valuable) are used before monthly credits (which reset/expire).

## API Response Changes

The credit balance API now returns:

```json
{
  "user_id": "user123",
  "credits_monthly": 100,
  "credits_permanent": 50,
  "balance": 150,  // Sum of both pools
  "total_earned": 200,
  "total_spent": 50,
  "last_updated": "2024-01-01T00:00:00Z"
}
```

## Transaction Tracking

Credit transactions now include pool information:

```json
{
  "transaction_id": "txn_123",
  "user_id": "user123",
  "type": "spent",
  "pool": "permanent",  // or "monthly"
  "amount": -10,
  "description": "Story generation (from permanent pool)",
  "created_at": "2024-01-01T00:00:00Z"
}
```

## Backward Compatibility

The system maintains backward compatibility by:
- Providing a computed `balance` property that sums both pools
- Auto-determining the appropriate pool when none is specified
- Gracefully handling existing records through automatic migration

## Future Considerations

- Monthly credit reset logic (when subscriptions renew)
- Credit expiration policies for monthly credits
- Admin tools for manual credit adjustments
- Analytics and reporting on credit pool usage 