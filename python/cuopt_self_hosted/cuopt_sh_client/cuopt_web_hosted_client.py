# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import warnings
from typing import Dict, Optional, Union
from urllib.parse import urlparse, urljoin

import requests

from .cuopt_self_host_client import CuOptServiceSelfHostClient, mime_type

log = logging.getLogger(__name__)


class CuOptServiceWebHostedClient(CuOptServiceSelfHostClient):
    """
    Web-hosted version of the CuOptServiceClient that supports endpoint URLs
    and authentication mechanisms for cloud-hosted services.
    
    This client is specifically designed for web-hosted cuOpt services and
    requires an endpoint URL. For self-hosted services with ip/port parameters,
    use CuOptServiceSelfHostClient instead.
    
    Parameters
    ----------
    endpoint : str
        Full endpoint URL for the cuOpt service. Required parameter. Examples:
        - "https://api.nvidia.com/cuopt/v1"
        - "https://inference.nvidia.com/cuopt"
        - "http://my-cuopt-service.com:8080/api"
    api_key : str, optional
        API key for authentication. Will be sent as Bearer token in Authorization header.
        Can also be set via CUOPT_API_KEY environment variable.
    base_path : str, optional
        Base path to append to the endpoint if not included in endpoint URL.
        Defaults to "/cuopt" if not specified in endpoint.
    self_signed_cert : str, optional
        Path to self-signed certificate for HTTPS connections.
    **kwargs
        Additional parameters passed to parent CuOptServiceSelfHostClient
        (excluding ip, port, use_https which are determined from endpoint)
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        base_path: Optional[str] = None,
        self_signed_cert: str = "",
        **kwargs
    ):
        if not endpoint:
            raise ValueError("endpoint parameter is required for CuOptServiceWebHostedClient")
            
        # Handle authentication from environment variables
        self.api_key = api_key or os.getenv("CUOPT_API_KEY")
        
        # Parse endpoint URL
        self._parsed_endpoint = self._parse_endpoint_url(endpoint, base_path)
        
        # Extract connection parameters from endpoint
        ip = self._parsed_endpoint["host"]
        port = str(self._parsed_endpoint["port"]) if self._parsed_endpoint["port"] else ""
        use_https = self._parsed_endpoint["scheme"] == "https"
        self._base_path = self._parsed_endpoint["path"]
            
        # Initialize parent class with extracted parameters
        super().__init__(
            ip=ip,
            port=port,
            use_https=use_https,
            self_signed_cert=self_signed_cert,
            **kwargs
        )
        
        # Override URL construction with endpoint-based URLs
        self._construct_endpoint_urls()
            
    def _parse_endpoint_url(self, endpoint: str, base_path: Optional[str] = None) -> Dict[str, Union[str, int, None]]:
        """
        Parse endpoint URL and extract components.
        
        Parameters
        ----------
        endpoint : str
            Full endpoint URL
        base_path : str, optional
            Base path to use if not included in endpoint
            
        Returns
        -------
        dict
            Parsed URL components
        """
        # Add protocol if missing
        if not endpoint.startswith(("http://", "https://")):
            log.warning(f"No protocol specified in endpoint '{endpoint}', assuming https://")
            endpoint = f"https://{endpoint}"
            
        parsed = urlparse(endpoint)
        
        if not parsed.hostname:
            raise ValueError(f"Invalid endpoint URL: {endpoint}")
            
        # Determine base path
        path = parsed.path.rstrip("/")
        if not path and base_path:
            path = base_path.rstrip("/")
        elif not path:
            path = "/cuopt"
            
        return {
            "scheme": parsed.scheme,
            "host": parsed.hostname,
            "port": parsed.port,
            "path": path,
            "full_url": f"{parsed.scheme}://{parsed.netloc}{path}"
        }
        
    def _construct_endpoint_urls(self):
        """Construct service URLs from parsed endpoint.
        
        For web-hosted cuOpt services (like NVIDIA's API), the endpoint
        URL is typically the complete service endpoint that handles all
        operations. Unlike self-hosted services that have separate paths
        for /request, /log, /solution, web-hosted APIs often use a single
        endpoint for all operations.
        """
        base_url = self._parsed_endpoint["full_url"]
        
        # For web-hosted services, use the provided endpoint directly
        # This matches the curl example which POSTs directly to the endpoint
        self.request_url = base_url
        
        # Log and solution URLs may still follow traditional patterns
        # but many web-hosted APIs use the same endpoint for all operations
        self.log_url = urljoin(base_url + "/", "log")
        self.solution_url = urljoin(base_url + "/", "solution")
        
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers."""
        headers = {}
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        return headers
        
    def _make_http_request(self, method: str, url: str, **kwargs):
        """
        Override parent method to add authentication headers and handle auth errors.
        
        Parameters
        ----------
        method : str
            HTTP method (GET, POST, DELETE, etc.)
        url : str
            Request URL
        **kwargs
            Additional arguments passed to requests.request()
            
        Returns
        -------
        requests.Response
            HTTP response object
        """
        # Add authentication headers
        headers = kwargs.get("headers", {})
        headers.update(self._get_auth_headers())
        kwargs["headers"] = headers
        
        # Make request
        response = requests.request(method, url, **kwargs)
        
        # Handle authentication errors
        if response.status_code == 401:
            raise ValueError("Authentication failed. Please check your API key.")
        elif response.status_code == 403:
            raise ValueError("Access forbidden. Please check your permissions.")
            
        return response


def create_client(
    endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> Union[CuOptServiceWebHostedClient, CuOptServiceSelfHostClient]:
    """
    Factory function to create appropriate client based on parameters.
    
    Creates CuOptServiceWebHostedClient if endpoint is provided, otherwise
    creates CuOptServiceSelfHostClient for legacy ip/port usage.
    
    Parameters
    ----------
    endpoint : str, optional
        Full endpoint URL. If provided, creates a web-hosted client.
        Required for web-hosted client creation.
    api_key : str, optional
        API key for web-hosted client authentication. Will be sent as Bearer token.
    **kwargs
        Additional parameters passed to the selected client
        
    Returns
    -------
    CuOptServiceWebHostedClient or CuOptServiceSelfHostClient
        Web-hosted client if endpoint provided, self-hosted client otherwise
        
    Examples
    --------
    # Creates web-hosted client
    client = create_client(
        endpoint="https://api.nvidia.com/cuopt/v1",
        api_key="your-key"
    )
    
    # Creates self-hosted client
    client = create_client(ip="192.168.1.100", port="5000")
    """
    if endpoint:
        # Create web-hosted client - endpoint is required
        return CuOptServiceWebHostedClient(
            endpoint=endpoint,
            api_key=api_key,
            **kwargs
        )
    elif api_key:
        # Authentication provided but no endpoint - this is an error
        raise ValueError(
            "api_key provided but no endpoint specified. "
            "Web-hosted client requires an endpoint URL. "
            "Use CuOptServiceSelfHostClient for ip/port connections."
        )
    else:
        # Create self-hosted client for legacy ip/port usage
        return CuOptServiceSelfHostClient(**kwargs)
