# Stripe Product Setup Guide

This guide explains how to set up your Stripe products to work seamlessly with the real-time product fetching system.

## Quick Setup Steps

### 1. Create Products in Stripe Dashboard

1. Go to [Stripe Dashboard > Products](https://dashboard.stripe.com/products)
2. Click "Add product"
3. Fill in product details and **add metadata** for automatic categorization

### 2. Product Metadata for Auto-Classification

Add these metadata keys to your Stripe products for automatic categorization:

#### Product Types

| Metadata Key | Value | Description |
|--------------|-------|-------------|
| `product_type` | `SUBSCRIPTION` | For recurring subscription plans |
| `product_type` | `BONUS` | For one-time credit purchases |

#### Credit Configuration

| Metadata Key | Value | Description | Used For |
|--------------|-------|-------------|-----------|
| `credits_permanent` | Number (e.g., `100`) | Credits granted for one-time purchases | Bonus products |
| `credits_monthly` | Number (e.g., `300`) | Monthly credit allocation | Subscription products |

**Automatic Detection**: The system primarily uses Stripe's price type for classification:
- **Recurring prices** → `SUBSCRIPTION` type
- **One-time prices** → `BONUS` type

**Metadata Override**: Product or price metadata with `product_type` will override automatic detection.

**Fallback Logic**: If price type is unavailable, the system uses product name patterns:
- Products with "subscription", "monthly", or "plan" in the name → `SUBSCRIPTION`
- Everything else → `BONUS`

### 3. Example Stripe Products

#### Subscription Product
- **Name**: "Premium Monthly Plan"
- **Description**: "Monthly subscription with 300 AI credits"
- **Price**: $9.99/month (recurring)
- **Price Metadata**: `credits_monthly: 300`
- **Type**: Automatically detected as `SUBSCRIPTION` (recurring price)

#### Bonus Credits Product
- **Name**: "100 AI Credits"
- **Description**: "One-time purchase of AI credits"
- **Price**: $4.99 (one-time)
- **Price Metadata**: `credits_permanent: 100`
- **Type**: Automatically detected as `BONUS` (one-time price)

#### Advanced Configuration Example
- **Product Metadata**: `product_type: BONUS` (explicit override)
- **Price Metadata**: `credits_permanent: 250` (credits to grant)

## Real-Time Product Fetching

The system automatically fetches products from Stripe in real-time when:
- Users visit the payment screen
- Payment intents are created

**No manual syncing required!** Products are always up-to-date with your Stripe catalog.

The system supports both one-time and recurring prices, automatically categorizing them based on Stripe's price type.

## Benefits of Real-Time Fetching

✅ **Always Current** - Products are always in sync with Stripe  
✅ **No Manual Work** - No scripts to run or caches to manage  
✅ **Instant Updates** - Changes in Stripe appear immediately  
✅ **Simplified Architecture** - Fewer components to maintain  
✅ **Smart Classification** - Automatic product type detection based on price type

## Web-Only Payment Implementation

**Current Implementation**: Payments are only available on the web version of the app.

### Why Web-Only?

- **Stripe Checkout** - Proven, secure, and optimized payment experience
- **No Complex SDK Setup** - Works with any web framework out of the box
- **Mobile Responsive** - Stripe Checkout works perfectly on mobile browsers
- **Faster Implementation** - Single platform reduces complexity
- **Future Expansion** - Easy to add mobile payments later if needed

### How It Works

1. **Web Users**: Redirected to Stripe's hosted checkout page
2. **Mobile Users**: Shown a friendly message directing them to use the web version
3. **Payment Processing**: All payments handled through Stripe Checkout sessions
4. **Webhooks**: Single webhook handler for `checkout.session.completed` events

### API Implementation

**Single Endpoint**:
- `POST /payment/create-checkout-session` - Creates checkout session for web users

**Webhook Events**:
- `checkout.session.completed` - Handles successful payments and subscriptions

### User Experience

#### Web Experience
```
Select Product → Click "Continue to Checkout" → Redirect to Stripe → Complete Payment → Return to App
```

#### Mobile Experience  
```
Tap Payment → See "Web Only" Message → Get Instructions to Use Browser
```

### Technical Benefits

🚀 **Faster Development** - Single payment flow to maintain  
🔒 **Maximum Security** - Stripe handles all sensitive data  
📱 **Mobile Compatible** - Works in mobile browsers  
🎨 **Consistent UX** - Stripe's optimized checkout experience  
⚡ **Easy Testing** - Simple webhook and redirect flow  

## Credit System Integration

### Automatic Credit Management

The system automatically handles credit allocation based on product type:

#### For Subscription Products
- **Monthly Credits**: Granted based on `credits_monthly` metadata
- **Renewal Handling**: Credits reset each billing period
- **Plan Changes**: Credit allocation adjusts automatically

#### For Bonus Products
- **Permanent Credits**: Added to user's account based on `credits_permanent` metadata
- **Immediate Availability**: Credits are available right after purchase
- **Quantity Support**: Supports multiple quantities in a single purchase

### Credit Calculation Functions

The system includes specialized functions for credit calculation:

- `get_credits_for_bonus_purchase(product_id, quantity)` - Calculates credits for one-time purchases
- `get_credits_for_subscription_purchase(plan_name)` - Determines monthly credit allocation for subscriptions

### Fallback Credit Values

If metadata is not configured, the system uses fallback calculations:

#### Bonus Products
- **Primary**: 1 credit per dollar spent
- **Fallback**: 10 credits per item

#### Subscription Products
Default monthly credits by plan name:
- `basic`: 100 credits
- `premium`: 300 credits  
- `pro`: 500 credits
- `starter`: 50 credits
- **Default**: 100 credits

## Local Development & Webhook Testing

### Option 1: Stripe CLI (Recommended)

The Stripe CLI is the official and best way to test webhooks locally.

#### Installation

**macOS:**
```bash
brew install stripe/stripe-cli/stripe
```

**Linux/Windows:** Follow instructions at [stripe.com/docs/stripe-cli](https://stripe.com/docs/stripe-cli)

#### Setup & Testing

1. **Login to Stripe:**
   ```bash
   stripe login
   ```
   This opens a browser to authenticate with your Stripe account.

2. **Start Your Local API:**
   ```bash
   # Terminal 1: Start FastAPI server
   uvicorn app.server.main:app --reload --host 0.0.0.0 --port 8080
   ```

3. **Forward Webhooks to Local Server:**
   ```bash
   # Terminal 2: Forward Stripe webhooks to your local API
   stripe listen --forward-to localhost:8080/payment/webhook
   ```

4. **Copy Webhook Secret:**
   The `stripe listen` command outputs:
   ```
   > Ready! Your webhook signing secret is whsec_1234abcd... (^C to quit)
   ```
   Copy this secret and add it to your environment:
   ```bash
   export BUCHAI_STRIPE_WEBHOOK_SECRET="whsec_1234abcd..."
   ```

5. **Test Webhook Events:**
   ```bash
   # Terminal 3: Trigger test webhook events
   stripe trigger checkout.session.completed
   stripe trigger invoice.payment_succeeded
   stripe trigger customer.subscription.created
   ```

#### Benefits of Stripe CLI:
- ✅ **Official Stripe tool** - Always up-to-date
- ✅ **No external dependencies** - No need for ngrok accounts  
- ✅ **Easy testing** - Built-in event triggering
- ✅ **Real-time logs** - See webhook events as they happen
- ✅ **Automatic cleanup** - No manual webhook management

### Option 2: ngrok (Alternative)

If you prefer using ngrok:

#### Installation
```bash
brew install ngrok
```

#### Setup
1. **Start Your API:**
   ```bash
   uvicorn app.server.main:app --reload --host 0.0.0.0 --port 8080
   ```

2. **Expose Local Server:**
   ```bash
   # In another terminal
   ngrok http 8080
   ```

3. **Configure Stripe Dashboard:**
   - Copy the ngrok URL (e.g., `https://abc123.ngrok.io`)
   - Go to [Stripe Dashboard > Webhooks](https://dashboard.stripe.com/webhooks)
   - Add endpoint: `https://abc123.ngrok.io/payment/webhook`
   - Select events: `checkout.session.completed`, `invoice.payment_succeeded`, `customer.subscription.created`
   - Copy the webhook signing secret to your environment

### Development Environment Setup

Create a `.env` file in your project root:

```env
# Development Environment
BUCHAI_ENV=d

# Authentication
BUCHAI_AUTH_JWT_KEY=your_jwt_secret_key_here

# Stripe Configuration (TEST MODE)
BUCHAI_STRIPE_SECRET_KEY=sk_test_your_test_secret_key_here
BUCHAI_STRIPE_WEBHOOK_SECRET=whsec_from_stripe_cli_or_dashboard

# Frontend (.env in Buch-AI-App/)
EXPO_PUBLIC_STRIPE_PUBLISHABLE_KEY=pk_test_your_test_publishable_key_here
```

### Complete End-to-End Testing

1. **Start Development Environment:**
   ```bash
   # Terminal 1: API with hot reload
   uvicorn app.server.main:app --reload --host 0.0.0.0 --port 8080
   
   # Terminal 2: Webhook forwarding
   stripe listen --forward-to localhost:8080/payment/webhook
   
   # Terminal 3: React Native app
   cd Buch-AI-App && npm start
   ```

2. **Test Payment Flow:**
   - Navigate to payment screen in your app
   - Select a product (both one-time and subscription)
   - Use test card: `4242 4242 4242 4242`
   - Complete payment
   - Watch webhook events in Terminal 2
   - Check database for payment records and credit allocation

3. **Manual Webhook Testing:**
   ```bash
   # Test different scenarios
   stripe trigger checkout.session.completed
   stripe trigger invoice.payment_succeeded
   stripe trigger customer.subscription.created
   stripe trigger customer.subscription.updated
   ```

4. **Check Database:**
   ```sql
   -- Check payment records
   SELECT * FROM `bai-buchai-p.payments.records` 
   ORDER BY created_at DESC 
   LIMIT 10;
   
   -- Check subscription records
   SELECT * FROM `bai-buchai-p.users.subscriptions`
   ORDER BY created_at DESC
   LIMIT 10;
   
   -- Check credit transactions
   SELECT * FROM `bai-buchai-p.users.credit_transactions`
   ORDER BY created_at DESC
   LIMIT 10;
   ```

## Production Webhook Setup

Configure Stripe webhooks for production deployment:

1. Go to [Stripe Dashboard > Webhooks](https://dashboard.stripe.com/webhooks)
2. Add endpoint: `https://your-api-url/payment/webhook`
3. Select events:
   - `checkout.session.completed` (for all payments)
   - `invoice.payment_succeeded` (for subscription renewals)
   - `customer.subscription.created` (for new subscriptions)
   - `customer.subscription.updated` (for subscription changes)
   - `customer.subscription.deleted` (for cancellations)
4. Copy webhook signing secret to your production environment variables

## Environment Variables

Ensure these are set in your environment:

### Backend Environment Variables
```env
# Stripe Configuration
BUCHAI_STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key_here  # or sk_live_ for production
BUCHAI_STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret_here

# Other required variables
BUCHAI_ENV=d  # or 'p' for production
BUCHAI_AUTH_JWT_KEY=your_jwt_secret_key
BUCHAI_HF_API_KEY=your_huggingface_api_key
```

### Frontend Environment Variables
**Note**: No Stripe environment variables needed for web-only checkout!

## Testing

Use Stripe's test card numbers for different scenarios:

| Card Number | Scenario |
|-------------|----------|
| `4242 4242 4242 4242` | Successful payment |
| `4000 0000 0000 0002` | Declined payment |
| `4000 0025 0000 3155` | Requires 3D Secure authentication |
| `4000 0000 0000 9995` | Insufficient funds |
| `4000 0000 0000 9987` | Lost card |

## Product Requirements

For products to appear in your app, they must:
1. **Be active** in Stripe
2. **Have at least one active price**
3. **Support both one-time and recurring pricing**

### Supported Price Types
- ✅ **One-time prices** - Automatically classified as `BONUS`
- ✅ **Recurring prices** - Automatically classified as `SUBSCRIPTION`
- ✅ **Mixed pricing** - Same product can have both types

## Benefits of Enhanced Product System

✅ **Smart Classification** - Automatic type detection based on price type  
✅ **Always Current** - Products are always in sync with Stripe  
✅ **No Manual Work** - No scripts to run or caches to manage  
✅ **Instant Updates** - Changes in Stripe appear immediately  
✅ **Flexible Credit System** - Supports both permanent and monthly credits  
✅ **Subscription Support** - Full subscription lifecycle management  

## Troubleshooting

### No Products Showing
1. Check Stripe products are **active**
2. Verify products have **active prices**
3. Check environment variables are set correctly
4. Review API logs for Stripe errors

### Credit Allocation Issues
1. **Bonus Credits Not Granted:**
   - Verify `credits_permanent` metadata is set on price or product
   - Check webhook processing logs
   - Ensure price type is `one_time`

2. **Subscription Credits Not Allocated:**
   - Verify `credits_monthly` metadata is set
   - Check price nickname matches plan name
   - Ensure price type is `recurring`

3. **Wrong Credit Amount:**
   - Check metadata values are valid integers
   - Review fallback credit calculation logic
   - Verify price type detection

### Webhook Issues (Development)
1. **Webhooks not received:**
   - Verify your API is running on port 8080
   - Check `stripe listen` is forwarding to correct URL
   - Ensure webhook secret matches environment variable

2. **Authentication errors:**
   - Run `stripe login` to re-authenticate
   - Verify Stripe keys are in test mode for development

3. **Connection issues:**
   - Check firewall settings
   - Verify network connectivity
   - Try restarting `stripe listen`

### Payment Failures
1. Verify webhook endpoint is accessible
2. Check webhook signing secret
3. Review application logs
4. Test with Stripe test cards

### Product Type Issues
1. Add explicit `product_type` metadata to Stripe products or prices
2. Check that price types are correctly set in Stripe
3. Review the automatic classification logic in `determine_product_type`
4. Verify product names contain expected keywords for fallback logic

## Security Best Practices

### Development
- ✅ Always use test keys (`sk_test_`, `pk_test_`) in development
- ✅ Never commit API keys to version control
- ✅ Use environment variables for all secrets
- ✅ Verify webhook signatures in your endpoint

### Production
- ✅ Use live keys (`sk_live_`, `pk_live_`) only in production
- ✅ Enable webhook signature verification
- ✅ Use HTTPS for all webhook endpoints
- ✅ Monitor webhook delivery and retry failed events
- ✅ Implement proper error handling and logging 