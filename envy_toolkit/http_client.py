"""
Common HTTP client utility for the Envy Zeitgeist Engine.

This module provides a unified HTTP client interface that eliminates
duplicate aiohttp patterns across collectors and other components.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

import aiohttp
import feedparser
from bs4 import BeautifulSoup

from .config import get_api_config
from .rate_limiter import rate_limiter_registry

logger = logging.getLogger(__name__)


class HTTPResponse:
    """Wrapper for HTTP response data."""

    def __init__(
        self,
        status: int,
        text: str,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        url: str = ""
    ):
        self.status = status
        self.text = text
        self.json_data = json_data
        self.headers = headers or {}
        self.url = url
        self.success = 200 <= status < 300

    def json(self) -> Dict[str, Any]:
        """Get JSON data from response."""
        if self.json_data is None:
            raise ValueError("Response does not contain JSON data")
        return self.json_data

    def soup(self) -> BeautifulSoup:
        """Get BeautifulSoup object from HTML response."""
        return BeautifulSoup(self.text, 'html.parser')

    def feed(self) -> Any:
        """Parse RSS/Atom feed from response."""
        return feedparser.parse(self.text)


class HTTPClient:
    """Common HTTP client with consistent timeout, error handling, and rate limiting."""

    def __init__(
        self,
        service_name: str = "default",
        session: Optional[aiohttp.ClientSession] = None,
        default_timeout: float = 10.0,
        user_agent: str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    ):
        """Initialize HTTP client.

        Args:
            service_name: Name of service for configuration lookup
            session: Optional existing aiohttp session
            default_timeout: Default timeout in seconds
            user_agent: User agent string for requests
        """
        self.service_name = service_name
        self.default_timeout = default_timeout
        self.user_agent = user_agent
        self._session = session
        self._session_created = False

        # Get configuration if available
        try:
            from .config import APIConfig
            self.config: Optional[APIConfig] = get_api_config(service_name)
        except (ValueError, KeyError):
            # Use defaults if no config found
            self.config = None

        # Get rate limiter if available
        try:
            self.rate_limiter = rate_limiter_registry.get(service_name)
        except (ValueError, KeyError):
            self.rate_limiter = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.default_timeout)
            if self.config:
                timeout = aiohttp.ClientTimeout(
                    total=self.config.timeout.total_timeout,
                    connect=self.config.timeout.connect_timeout,
                    sock_read=self.config.timeout.read_timeout
                )

            headers = {"User-Agent": self.user_agent}
            if self.config and self.config.api_key:
                headers.update(self.config.get_auth_headers())

            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=timeout
            )
            self._session_created = True

        return self._session

    async def close(self) -> None:
        """Close the HTTP session if we created it."""
        if self._session_created and self._session:
            await self._session.close()
            self._session = None
            self._session_created = False

    async def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        allow_redirects: bool = True
    ) -> HTTPResponse:
        """Make GET request with common error handling.

        Args:
            url: URL to request
            params: Query parameters
            headers: Additional headers
            timeout: Request timeout (overrides default)
            allow_redirects: Whether to follow redirects

        Returns:
            HTTPResponse object with response data

        Raises:
            aiohttp.ClientError: For HTTP errors
            asyncio.TimeoutError: For timeout errors
        """
        return await self._request(
            "GET", url, params=params, headers=headers,
            timeout=timeout, allow_redirects=allow_redirects
        )

    async def post(
        self,
        url: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None
    ) -> HTTPResponse:
        """Make POST request with common error handling.

        Args:
            url: URL to request
            json: JSON data to send
            data: Form data to send
            headers: Additional headers
            timeout: Request timeout (overrides default)

        Returns:
            HTTPResponse object with response data

        Raises:
            aiohttp.ClientError: For HTTP errors
            asyncio.TimeoutError: For timeout errors
        """
        return await self._request(
            "POST", url, json=json, data=data,
            headers=headers, timeout=timeout
        )

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any
    ) -> HTTPResponse:
        """Make HTTP request with rate limiting and error handling."""
        # Apply rate limiting if configured
        if self.rate_limiter:
            await self.rate_limiter.acquire()

        session = await self._get_session()

        # Set timeout
        if "timeout" in kwargs and kwargs["timeout"] is not None:
            kwargs["timeout"] = aiohttp.ClientTimeout(total=kwargs["timeout"])
        elif not kwargs.get("timeout"):
            # Use default timeout if none specified
            kwargs["timeout"] = aiohttp.ClientTimeout(total=self.default_timeout)

        try:
            async with session.request(method, url, **kwargs) as response:
                text = await response.text()

                # Try to parse JSON if content type suggests it
                json_data = None
                content_type = response.headers.get("content-type", "").lower()
                if "application/json" in content_type:
                    try:
                        json_data = await response.json()
                    except Exception:
                        # JSON parsing failed, that's ok
                        pass

                return HTTPResponse(
                    status=response.status,
                    text=text,
                    json_data=json_data,
                    headers=dict(response.headers),
                    url=str(response.url)
                )

        except Exception as e:
            logger.error(f"HTTP request failed for {url}: {e}")
            raise

    async def get_multiple(
        self,
        urls: List[str],
        max_concurrent: int = 5,
        **kwargs: Any
    ) -> List[HTTPResponse]:
        """Make multiple GET requests concurrently.

        Args:
            urls: List of URLs to request
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional arguments passed to get()

        Returns:
            List of HTTPResponse objects (may contain exceptions)
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_one(url: str) -> HTTPResponse:
            async with semaphore:
                try:
                    return await self.get(url, **kwargs)
                except Exception as e:
                    logger.error(f"Failed to fetch {url}: {e}")
                    # Return error response
                    return HTTPResponse(
                        status=0,
                        text=str(e),
                        url=url
                    )

        tasks = [fetch_one(url) for url in urls]
        return await asyncio.gather(*tasks, return_exceptions=False)


@asynccontextmanager
async def http_client(
    service_name: str = "default",
    session: Optional[aiohttp.ClientSession] = None,
    **kwargs: Any
) -> AsyncIterator[HTTPClient]:
    """Context manager for HTTP client.

    Args:
        service_name: Name of service for configuration lookup
        session: Optional existing aiohttp session
        **kwargs: Additional arguments for HTTPClient

    Yields:
        HTTPClient instance
    """
    client = HTTPClient(service_name=service_name, session=session, **kwargs)
    try:
        yield client
    finally:
        await client.close()


# Convenience functions for common patterns
async def fetch_rss_feed(
    url: str,
    service_name: str = "default",
    timeout: float = 10.0
) -> Any:
    """Fetch and parse RSS feed.

    Args:
        url: RSS feed URL
        service_name: Service name for configuration
        timeout: Request timeout

    Returns:
        Parsed feedparser feed object

    Raises:
        Exception: If fetch or parse fails
    """
    async with http_client(service_name) as client:
        response = await client.get(url, timeout=timeout)
        if response.success:
            return response.feed()
        else:
            raise Exception(f"Failed to fetch RSS feed: HTTP {response.status}")


async def fetch_html(
    url: str,
    service_name: str = "default",
    timeout: float = 10.0
) -> BeautifulSoup:
    """Fetch and parse HTML content.

    Args:
        url: HTML page URL
        service_name: Service name for configuration
        timeout: Request timeout

    Returns:
        BeautifulSoup object

    Raises:
        Exception: If fetch or parse fails
    """
    async with http_client(service_name) as client:
        response = await client.get(url, timeout=timeout)
        if response.success:
            return response.soup()
        else:
            raise Exception(f"Failed to fetch HTML: HTTP {response.status}")


async def fetch_json(
    url: str,
    service_name: str = "default",
    timeout: float = 10.0,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Fetch and parse JSON content.

    Args:
        url: JSON API URL
        service_name: Service name for configuration
        timeout: Request timeout
        params: Query parameters
        headers: Additional headers

    Returns:
        Parsed JSON data

    Raises:
        Exception: If fetch or parse fails
    """
    async with http_client(service_name) as client:
        response = await client.get(url, params=params, headers=headers, timeout=timeout)
        if response.success:
            return response.json()
        else:
            raise Exception(f"Failed to fetch JSON: HTTP {response.status}")
