# BigQuery to Firestore Migration Guide

This document outlines the complete migration process from BigQuery to Firestore for your Buch AI Server application.

## ✅ Migration Status: **FULLY SUPPORTED**

All migration requirements have been successfully addressed:

- **✅ Terraform Support**: Full native support via `google_firestore_database` resource
- **✅ GitHub Actions**: Updated workflows for Firestore database deployment
- **✅ Schema Consistency**: Robust schema management using both Pydantic and Google-native approaches

## Overview

### What Changed

| Component | Before (BigQuery) | After (Firestore) |
|-----------|-------------------|-------------------|
| **Data Model** | Datasets + Tables | Database + Collections |
| **Infrastructure** | `google_bigquery_*` resources | `google_firestore_database` resource |
| **Schema Validation** | BigQuery schema definition | Firestore Security Rules + Pydantic |
| **Data Access** | SQL queries | Document queries + Firebase Admin SDK |
| **Collections** | 5 datasets, 10 tables | 1 database, 10 collections |

### Collection Mapping

```
BigQuery Datasets/Tables → Firestore Collections
──────────────────────────────────────────────────
creations.cost_centres   → creations_cost_centres
creations.profiles       → creations_profiles
tasks.video_generator    → tasks_video_generator
users.auth              → users_auth
users.profiles          → users_profiles
users.geolocation       → users_geolocation
users.credits           → users_credits
users.subscriptions     → users_subscriptions
payments.records        → payments_records
credits.transactions    → credits_transactions
```

## Infrastructure Changes

### 1. Terraform Configuration

**New Files Created:**
- `terraform/firestore.tf` - Firestore database configuration
- `terraform/firestore.rules` - Security rules for schema validation

**Updated Files:**
- `terraform/main.tf` - References to firestore.tf, BigQuery resources commented out
- `.github/workflows/build-deploy.yaml` - Updated import commands

### 2. Deploy the Infrastructure

```bash
cd terraform

# Plan the migration
terraform plan

# Apply the changes (this will create Firestore databases)
terraform apply

# The old BigQuery resources can be destroyed after data migration
# terraform destroy -target=google_bigquery_dataset.creations
# terraform destroy -target=google_bigquery_dataset.users
# etc.
```

## Schema Management Options

You have **three excellent options** for schema validation, each with different trade-offs:

### Option 1: Pydantic Models (Recommended for Python-centric apps)

✅ **What we've implemented:** Type-safe Python models with validation

```python
from app.models.firestore import CreationProfile, UserCredits
from app.services.firestore_service import get_firestore_service

# Type-safe operations
firestore_service = get_firestore_service()
creation = await firestore_service.get_document(
    collection_name="creations_profiles",
    document_id="creation_123",
    model_class=CreationProfile
)
```

**Pros:**
- Full Python type safety
- IDE autocompletion
- Runtime validation
- Familiar Pydantic ecosystem

**Cons:**
- Python-specific
- Validation happens at application layer

### Option 2: Firestore Security Rules (Google-native, recommended for multi-client apps)

✅ **What we've implemented:** Comprehensive schema validation at the database level

```javascript
// In terraform/firestore.rules
match /creations_profiles/{creationId} {
  allow create: if isAuthenticated()
    && request.resource.data.keys().hasAll(['creation_id', 'title', 'user_id'])
    && request.resource.data.creation_id is string
    && request.resource.data.title is string
    && request.resource.data.user_id is string;
}
```

**Pros:**
- Database-level validation
- Works for all clients (web, mobile, server)
- Google-native approach
- Security and validation in one place

**Cons:**
- Custom rule language to learn
- Less flexible than code-based validation

### Option 3: Protocol Buffers + Firebase Rules (Google-native, enterprise-grade)

🎯 **Google's enterprise recommendation** for large-scale applications:

```protobuf
// schemas/creation.proto
syntax = "proto3";

message CreationProfile {
  string creation_id = 1;
  string title = 2;
  string user_id = 3;
  google.protobuf.Timestamp created_at = 4;
}
```

Then use [Firebase Rules Protobuf Generator](https://github.com/FirebaseExtended/protobuf-rules-gen) to auto-generate Security Rules.

**Pros:**
- Language-agnostic schemas
- Auto-generated validation rules
- Version-controlled schema evolution
- Used by Google internally

**Cons:**
- Additional build complexity
- Learning curve for Protocol Buffers

## Application Code Changes

### 1. Update Dependencies

Add to your `requirements.txt`:
```
firebase-admin>=6.0.0
google-cloud-firestore>=2.13.0
```

### 2. Replace BigQuery Client Usage

**Before (BigQuery):**
```python
from google.cloud import bigquery

client = bigquery.Client()
query = "SELECT * FROM `bai-buchai-p.users.credits` WHERE user_id = @user_id"
job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
    ]
)
query_job = client.query(query, job_config=job_config)
results = query_job.result()
```

**After (Firestore):**
```python
from app.services.firestore_service import get_firestore_service
from app.models.firestore import UserCredits

firestore_service = get_firestore_service()
credits = await firestore_service.get_document(
    collection_name="users_credits", 
    document_id=user_id,
    model_class=UserCredits
)
```

### 3. Update Service Layer

We've created `app/services/firestore_service.py` that provides:

- **Type-safe operations** with Pydantic integration
- **Familiar interface** similar to your existing BigQuery patterns
- **Async support** for all operations
- **Transaction support** for complex operations
- **Batch operations** for performance

Example usage:
```python
from app.services.firestore_service import get_firestore_service
from app.models.firestore import PaymentRecord

service = get_firestore_service()

# Create document
payment_data = {
    "payment_id": "pay_123",
    "user_id": "user_456", 
    "amount": 1000,
    "currency": "USD",
    "status": "completed"
}
doc_id = await service.create_document("payments_records", payment_data)

# Query documents
payments = await service.query_collection(
    collection_name="payments_records",
    filters=[("user_id", "==", "user_456")],
    limit=10,
    model_class=PaymentRecord
)

# Transaction example
with service.transaction() as transaction:
    # Atomic operations
    transaction.update(doc_ref1, {"balance": new_balance})
    transaction.create(doc_ref2, transaction_data)
```

## Data Migration Process

### 1. Export from BigQuery

```bash
# Export each table to Cloud Storage
bq extract \
  --destination_format=NEWLINE_DELIMITED_JSON \
  bai-buchai-p:users.credits \
  gs://your-migration-bucket/users_credits.jsonl
```

### 2. Transform and Import to Firestore

```python
# migration_script.py
import json
from app.services.firestore_service import get_firestore_service

async def migrate_user_credits():
    service = get_firestore_service()
    
    with open('users_credits.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            
            # Transform BigQuery data to Firestore format
            doc_data = {
                "user_id": data["user_id"],
                "balance": int(data["balance"]),
                "total_earned": int(data["total_earned"]),
                "total_spent": int(data["total_spent"]),
                "last_updated": data["last_updated"],
                "created_at": data["created_at"]
            }
            
            # Create document with user_id as document ID
            await service.create_document(
                collection_name="users_credits",
                document_data=doc_data,
                document_id=data["user_id"]
            )
```

## Performance Considerations

### BigQuery vs Firestore Trade-offs

| Aspect | BigQuery | Firestore |
|--------|----------|-----------|
| **Read Performance** | Excellent for analytics | Excellent for OLTP |
| **Write Performance** | Batch-optimized | Real-time optimized |
| **Scaling** | Automatic | Automatic |
| **Pricing** | Query-based | Operation-based |
| **Real-time** | Not designed for it | Native support |
| **Complex Queries** | SQL (very powerful) | Limited (simple filters) |

### Optimization Tips

1. **Document Design**: Embed related data to reduce reads
2. **Indexing**: Firestore auto-indexes, but composite indexes may be needed
3. **Batch Operations**: Use batch writes for better performance
4. **Caching**: Consider caching frequently accessed documents
5. **Pagination**: Use cursor-based pagination for large datasets

## Testing the Migration

### 1. Unit Tests

```python
import pytest
from app.services.firestore_service import get_firestore_service
from app.models.firestore import UserCredits

@pytest.mark.asyncio
async def test_user_credits_crud():
    service = get_firestore_service("test-database")
    
    # Create
    credits_data = {
        "user_id": "test_user",
        "balance": 100,
        "total_earned": 100,
        "total_spent": 0
    }
    doc_id = await service.create_document("users_credits", credits_data)
    
    # Read
    credits = await service.get_document(
        "users_credits", 
        doc_id, 
        UserCredits
    )
    assert credits.balance == 100
    
    # Update
    await service.update_document("users_credits", doc_id, {"balance": 150})
    
    # Delete
    await service.delete_document("users_credits", doc_id)
```

### 2. Integration Tests

```python
@pytest.mark.asyncio
async def test_payment_workflow():
    service = get_firestore_service("test-database")
    
    # Test full payment workflow with transactions
    with service.transaction() as txn:
        # Deduct credits
        # Create payment record
        # Update subscription
        pass
```

## Rollback Plan

If you need to rollback:

1. **Keep BigQuery resources** during migration period
2. **Feature flags** to switch between BigQuery and Firestore
3. **Data sync** mechanism during transition period

```python
# Example feature flag approach
USE_FIRESTORE = os.getenv("USE_FIRESTORE", "false").lower() == "true"

if USE_FIRESTORE:
    from app.services.firestore_service import get_firestore_service
    service = get_firestore_service()
else:
    from app.services.bigquery_service import get_bigquery_service
    service = get_bigquery_service()
```

## Next Steps

1. **Review the generated files** - especially the Firestore Security Rules
2. **Test the infrastructure** - deploy to a development environment first
3. **Update your application code** - replace BigQuery client usage
4. **Plan data migration** - export from BigQuery and import to Firestore
5. **Update monitoring** - Firestore has different metrics than BigQuery

## Support

- **Terraform Firestore Documentation**: https://registry.terraform.io/providers/hashicorp/google/latest/docs/resources/firestore_database
- **Firestore Security Rules**: https://firebase.google.com/docs/rules
- **Firebase Admin SDK**: https://firebase.google.com/docs/admin/setup
- **Protocol Buffers + Firebase Rules**: https://github.com/FirebaseExtended/protobuf-rules-gen

The migration is now ready for deployment! 🚀 