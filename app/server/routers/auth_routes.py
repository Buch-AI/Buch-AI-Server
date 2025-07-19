import logging
from datetime import datetime, timedelta, timezone
from traceback import format_exc
from typing import Annotated

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from google.cloud import bigquery
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel

from app.models.geolocation import GeolocationProcessor
from config import AUTH_JWT_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a router for authentication operations
auth_router = APIRouter()

# Google Cloud BigQuery client
client = bigquery.Client()

# JWT configuration
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    user_id: str | None = None
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


class UserInDB(User):
    hashed_password: str


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_password_hash(password):
    return pwd_context.hash(password)


def password_verified(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


async def get_user_db_record(username: str):
    query = """
    SELECT user_id, username, email, password_hash, is_active
    FROM `bai-buchai-p.users.auth`
    WHERE username = @username
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("username", "STRING", username)]
    )
    query_job = client.query(query, job_config=job_config)
    results = query_job.result()

    for row in results:
        return UserInDB(
            user_id=row.user_id,
            username=row.username,
            email=row.email,
            hashed_password=row.password_hash,
            disabled=not row.is_active,
        )
    return None


def get_client_ipv4(request: Request) -> str:
    """
    Extract the client's IP address from the request.

    Handles various proxy headers and fallbacks to direct connection.
    """
    # Check for forwarded headers (common in production behind load balancers)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can contain multiple IPs, take the first one
        return forwarded_for.split(",")[0].strip()

    # Check for real IP header
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    # Fallback to direct client connection
    client_host = request.client.host if request.client else "unknown"
    return client_host


async def authenticate_user(username: str, password: str):
    user = await get_user_db_record(username)
    if not user:
        return False
    if not password_verified(password, user.hashed_password):
        return False
    return user


async def _get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, AUTH_JWT_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            logger.error(f"No username found in token\n{format_exc()}")
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        logger.error(f"Invalid token error\n{format_exc()}")
        raise credentials_exception
    user = await get_user_db_record(token_data.username)
    if user is None:
        logger.error(f"User not found: {token_data.username}\n{format_exc()}")
        raise credentials_exception
    return user


async def get_current_user(
    current_user: Annotated[User, Depends(_get_current_user)],
):
    if current_user.disabled:
        logger.error(
            f"Inactive user attempted access: {current_user.username}\n{format_exc()}"
        )
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, AUTH_JWT_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def update_last_login(user_id: str) -> bool:
    """
    Update the last_login timestamp for a user in the database.

    Args:
        user_id: The ID of the user to update

    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        query = """
        UPDATE `bai-buchai-p.users.auth`
        SET last_login = CURRENT_TIMESTAMP()
        WHERE user_id = @user_id
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )
        query_job = client.query(query, job_config=job_config)
        query_job.result()  # Wait for the job to complete

        return True
    except Exception as e:
        logger.error(f"Failed to update last_login for user {user_id}: {str(e)}")
        return False


@auth_router.post("/token")
async def login_for_access_token(
    request: Request,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        logger.error(
            f"Failed login attempt for user: {form_data.username}\n{format_exc()}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update last login timestamp (non-blocking)
    if user.user_id:
        try:
            success = await update_last_login(user.user_id)
            if success:
                logger.info(f"Updated last_login for user: {user.username}")
            else:
                logger.warning(f"Failed to update last_login for user: {user.username}")
        except Exception as e:
            # Don't fail the login if last_login update fails
            logger.error(
                f"Error updating last_login for user {user.username}: {str(e)}"
            )

    # Log successful login geolocation data (non-blocking)
    try:
        client_ipv4 = get_client_ipv4(request)
        if client_ipv4 and client_ipv4 != "unknown" and user.user_id:
            processor = GeolocationProcessor(client_ipv4)
            success = processor.log_user(user.user_id)

            if success:
                logger.info(
                    f"Logged login geolocation for user {user.username} from IP {client_ipv4}"
                )
            else:
                logger.warning(
                    f"Failed to log login geolocation for user {user.username} from IP {client_ipv4}"
                )
        else:
            logger.warning(
                f"Skipping login geolocation logging - IP: {client_ipv4}, User ID: {user.user_id}"
            )

    except Exception as e:
        # Don't fail the login if geolocation logging fails
        logger.error(
            f"Error logging login geolocation for user {user.username}: {str(e)}"
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")


@auth_router.post("/token/refresh", response_model=Token)
async def refresh_access_token(
    current_user: Annotated[User, Depends(get_current_user)],
) -> Token:
    """
    Refresh an existing valid JWT token with a new expiration time.

    This endpoint allows users to obtain a new JWT token using their existing valid token,
    extending the session without requiring username/password authentication again.

    Args:
        current_user: User object from the validated JWT token

    Returns:
        Token: New JWT token with fresh expiration time

    Raises:
        HTTPException: If the current token is invalid or user is inactive
    """
    try:
        # Create a new access token with fresh expiration
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": current_user.username}, expires_delta=access_token_expires
        )

        # Update last login timestamp (non-blocking)
        if current_user.user_id:
            try:
                success = await update_last_login(current_user.user_id)
                if success:
                    logger.info(
                        f"Updated last_login for user during refresh: {current_user.username}"
                    )
                else:
                    logger.warning(
                        f"Failed to update last_login for user during refresh: {current_user.username}"
                    )
            except Exception as e:
                # Don't fail the refresh if last_login update fails
                logger.error(
                    f"Error updating last_login during refresh for user {current_user.username}: {str(e)}"
                )

        logger.info(f"Token refreshed successfully for user: {current_user.username}")

        return Token(access_token=access_token, token_type="bearer")

    except Exception as e:
        logger.error(
            f"Error refreshing token for user {current_user.username}: {str(e)}\n{format_exc()}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh token",
        )


@auth_router.get("/users/me", response_model=User)
async def get_users_me(
    current_user: Annotated[User, Depends(get_current_user)],
):
    """Get current user information."""
    return current_user
