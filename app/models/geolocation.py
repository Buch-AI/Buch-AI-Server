import logging
from ipaddress import AddressValueError
from ipaddress import ip_address as parse_ip_address
from traceback import format_exc
from typing import Optional

import requests
from fastapi import HTTPException, status
from pydantic import BaseModel, Field

from config import IPINFO_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeolocationResponse(BaseModel):
    """Pydantic response model for geolocation data from IPinfo.io API."""

    ip: str = Field(..., description="IP address that was queried")
    country_code: Optional[str] = Field(None, description="2-letter ISO country code")
    loc: Optional[str] = Field(None, description="Latitude,longitude coordinates")
    postal: Optional[str] = Field(None, description="Postal/ZIP code")
    timezone: Optional[str] = Field(None, description="Timezone identifier")
    org: Optional[str] = Field(None, description="Organization/ISP information")
    hostname: Optional[str] = Field(None, description="Hostname for the IP address")
    anycast: bool = Field(False, description="Whether the IP is anycast")
    bogon: bool = Field(False, description="Whether the IP is a bogon address")


class GeolocationProcessor:
    """
    Geolocation processor using IPinfo.io API.

    Makes a single API call on initialization and provides methods to access
    different parts of the geolocation data for the given IP address.
    """

    BASE_URL = "https://ipinfo.io"
    TIMEOUT = 10  # seconds

    def __init__(self, ip_address: str):
        """
        Initialize the processor with an IP address and fetch geolocation data.

        Args:
            ip_address: IP address to lookup

        Raises:
            HTTPException: If IP address is invalid or lookup fails
        """
        self.ip_address = ip_address.strip()
        self._response = self._fetch_geolocation_data()

    def _validate_ip_address(self, ip_address: str) -> bool:
        """
        Validate that the provided string is a valid IP address.

        Args:
            ip_address: IP address string to validate

        Returns:
            bool: True if valid IP address, False otherwise
        """
        try:
            ip_address_obj = parse_ip_address(ip_address.strip())
            # Check if it's a private, loopback, or multicast address
            if (
                ip_address_obj.is_private
                or ip_address_obj.is_loopback
                or ip_address_obj.is_multicast
            ):
                logger.warning(
                    f"IP address {ip_address} is private, loopback, or multicast"
                )
                return False
            return True
        except (AddressValueError, ValueError) as e:
            logger.error(f"Invalid IP address format: {ip_address}, error: {str(e)}")
            return False

    def _fetch_geolocation_data(self) -> GeolocationResponse:
        """
        Make HTTP request to IPinfo.io API for the initialized IP address.

        Returns:
            GeolocationResponse: Parsed response data

        Raises:
            HTTPException: If request fails or returns invalid data
        """
        # Validate IP address first
        if not self._validate_ip_address(self.ip_address):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid IP address: {self.ip_address}",
            )

        try:
            # Build URL
            url = f"{self.BASE_URL}/{self.ip_address}/json"

            # Add authentication if API key is available
            params = {}
            if IPINFO_API_KEY:
                params["token"] = IPINFO_API_KEY

            # Make request
            response = requests.get(
                url,
                params=params,
                timeout=self.TIMEOUT,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "BuchAI-Server/1.0",
                },
            )
            response.raise_for_status()

            # Parse response
            data = response.json()

            # Check if response indicates an error (bogon, invalid IP, etc.)
            if data.get("bogon"):
                logger.warning(f"IP address {self.ip_address} is a bogon address")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"IP address {self.ip_address} is a bogon (invalid/private) address",
                )

            # Map IPinfo's 'country' field to our 'country_code' field
            return GeolocationResponse(
                ip=data.get("ip", ""),
                country_code=data.get(
                    "country"
                ),  # IPinfo returns country code in 'country' field
                loc=data.get("loc"),
                postal=data.get("postal"),
                timezone=data.get("timezone"),
                org=data.get("org"),
                hostname=data.get("hostname"),
                anycast=data.get("anycast", False),
                bogon=data.get("bogon", False),
            )

        except requests.RequestException as e:
            logger.error(f"Error making request to IPinfo.io: {str(e)}\n{format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to fetch geolocation data: {str(e)}",
            )
        except ValueError as e:
            logger.error(f"Error parsing JSON response: {str(e)}\n{format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Invalid response from geolocation service",
            )
        except Exception as e:
            logger.error(
                f"Unexpected error in geolocation request: {str(e)}\n{format_exc()}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Unexpected error: {str(e)}",
            )

    def get_geolocation(self) -> str:
        """
        Get the basic geolocation information (country code).

        Returns:
            str: Country code or error message if not available
        """
        if self._response.country_code:
            return self._response.country_code
        else:
            return f"Location data not available for {self.ip_address}"

    def get_coords(self) -> tuple[str, str]:
        """
        Get the coordinates (latitude, longitude) for the IP address.

        Returns:
            Optional[tuple[str, str]]: Tuple of (latitude, longitude) as strings,
                                     or None if coordinates are not available
        """
        if self._response.loc:
            try:
                lat, lng = self._response.loc.split(",", 1)
                return (lat.strip(), lng.strip())
            except ValueError:
                logger.warning(f"Invalid coordinate format: {self._response.loc}")
                return (
                    f"Latitude not available for {self.ip_address}",
                    f"Longitude not available for {self.ip_address}",
                )
        else:
            return (
                f"Latitude not available for {self.ip_address}",
                f"Longitude not available for {self.ip_address}",
            )

    def get_country_code(self) -> str:
        """
        Get the country code for the IP address.

        Returns:
            str: 2-letter ISO country code (e.g., "US", "GB", "DE") or error message
        """
        if self._response.country_code:
            return self._response.country_code
        else:
            return f"Country code not available for {self.ip_address}"

    def log_user(self, user_id: str) -> bool:
        """
        Log user geolocation data to Firestore.

        Args:
            user_id: User ID to associate with the geolocation data

        Returns:
            bool: True if logging was successful, False otherwise
        """
        try:
            import uuid
            from datetime import datetime

            from app.services.firestore import get_firestore_service

            # Initialize Firestore service
            firestore_service = get_firestore_service()

            # Extract coordinates if available
            coord_lat = None
            coord_lon = None
            coords = self.get_coords()
            if coords:
                try:
                    coord_lat = float(coords[0])
                    coord_lon = float(coords[1])
                except (ValueError, IndexError):
                    logger.warning(f"Invalid coordinates for logging: {coords}")

            # Prepare the geolocation data
            geo_data = {
                "user_id": user_id,
                "time": datetime.utcnow(),
                "ipv4": self._response.ip,
                "geolocation": self.get_geolocation(),
                "coord_lat": coord_lat,
                "coord_lon": coord_lon,
                "country_code": self._response.country_code,
                "is_vpn": None,  # VPN detection not implemented yet
            }

            # Create document in Firestore
            import asyncio

            # Run async function in sync context
            async def _create_document():
                await firestore_service.create_document(
                    collection_name="users_geolocation",
                    document_data=geo_data,
                    document_id=str(uuid.uuid4()),
                )

            # Execute the async function
            asyncio.run(_create_document())

            logger.info(
                f"Successfully logged geolocation data for user {user_id} with IP {self.ip_address}"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to log geolocation data for user {user_id}: {str(e)}\n{format_exc()}"
            )
            return False
