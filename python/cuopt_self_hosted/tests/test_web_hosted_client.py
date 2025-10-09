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

import os
import pytest
import unittest.mock as mock
from unittest.mock import patch, MagicMock

from cuopt_sh_client import (
    CuOptServiceWebHostedClient,
    CuOptServiceSelfHostClient,
    create_client,
    mime_type,
)


class TestWebHostedClient:
    """Test suite for CuOptServiceWebHostedClient."""
    
    def test_endpoint_required(self):
        """Test that endpoint parameter is required."""
        with pytest.raises(ValueError, match="endpoint parameter is required"):
            CuOptServiceWebHostedClient()
            
        with pytest.raises(ValueError, match="endpoint parameter is required"):
            CuOptServiceWebHostedClient(endpoint="")
    
    def test_endpoint_url_parsing(self):
        """Test URL parsing functionality."""
        # Test basic HTTPS endpoint
        client = CuOptServiceWebHostedClient(endpoint="https://api.nvidia.com/cuopt/v1")
        assert client._parsed_endpoint["scheme"] == "https"
        assert client._parsed_endpoint["host"] == "api.nvidia.com"
        assert client._parsed_endpoint["port"] is None
        assert client._parsed_endpoint["path"] == "/cuopt/v1"
        
        # Test endpoint with port
        client = CuOptServiceWebHostedClient(endpoint="https://example.com:8080/api")
        assert client._parsed_endpoint["scheme"] == "https"
        assert client._parsed_endpoint["host"] == "example.com"
        assert client._parsed_endpoint["port"] == 8080
        assert client._parsed_endpoint["path"] == "/api"
        
        # Test endpoint without protocol (should default to https)
        with pytest.warns(UserWarning):
            client = CuOptServiceWebHostedClient(endpoint="inference.nvidia.com/cuopt")
        assert client._parsed_endpoint["scheme"] == "https"
        assert client._parsed_endpoint["host"] == "inference.nvidia.com"
        assert client._parsed_endpoint["path"] == "/cuopt"
        
        # Test endpoint without path (should default to /cuopt)
        client = CuOptServiceWebHostedClient(endpoint="https://example.com")
        assert client._parsed_endpoint["path"] == "/cuopt"
        
    def test_invalid_endpoint_url(self):
        """Test handling of invalid endpoint URLs."""
        with pytest.raises(ValueError, match="Invalid endpoint URL"):
            CuOptServiceWebHostedClient(endpoint="not-a-valid-url")
            
    def test_authentication_from_parameters(self):
        """Test authentication setup from parameters."""
        # Test API key (sent as Bearer token)
        client = CuOptServiceWebHostedClient(
            endpoint="https://api.nvidia.com/cuopt/v1",
            api_key="test-api-key"
        )
        headers = client._get_auth_headers()
        assert headers["Authorization"] == "Bearer test-api-key"
        
        # Test no authentication
        client = CuOptServiceWebHostedClient(endpoint="https://api.nvidia.com/cuopt/v1")
        headers = client._get_auth_headers()
        assert len(headers) == 0
        
    @patch.dict(os.environ, {"CUOPT_API_KEY": "env-api-key"})
    def test_authentication_from_environment(self):
        """Test authentication setup from environment variables."""
        client = CuOptServiceWebHostedClient(endpoint="https://api.nvidia.com/cuopt/v1")
        headers = client._get_auth_headers()
        assert headers["Authorization"] == "Bearer env-api-key"
        
    def test_parameter_precedence(self):
        """Test that parameters take precedence over environment variables."""
        with patch.dict(os.environ, {"CUOPT_API_KEY": "env-api-key"}):
            client = CuOptServiceWebHostedClient(
                endpoint="https://api.nvidia.com/cuopt/v1",
                api_key="param-api-key"
            )
            headers = client._get_auth_headers()
            assert headers["Authorization"] == "Bearer param-api-key"
        
    def test_url_construction_with_endpoint(self):
        """Test URL construction when endpoint is provided."""
        client = CuOptServiceWebHostedClient(endpoint="https://api.nvidia.com/cuopt/v1")
        assert client.request_url == "https://api.nvidia.com/cuopt/v1"
        assert client.log_url == "https://api.nvidia.com/cuopt/v1/log"
        assert client.solution_url == "https://api.nvidia.com/cuopt/v1/solution"
        
    def test_url_construction_from_endpoint_parsing(self):
        """Test URL construction from parsed endpoint components."""
        client = CuOptServiceWebHostedClient(endpoint="https://example.com:8080/custom")
        assert client.request_url == "https://example.com:8080/custom"
        assert client.log_url == "https://example.com:8080/custom/log"
        assert client.solution_url == "https://example.com:8080/custom/solution"
        
    @patch('cuopt_sh_client.cuopt_web_hosted_client.requests.request')
    def test_authenticated_request_with_api_key(self, mock_request):
        """Test that authenticated requests include API key."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_request.return_value = mock_response
        
        client = CuOptServiceWebHostedClient(
            endpoint="https://api.nvidia.com/cuopt/v1",
            api_key="test-api-key"
        )
        
        client._make_http_request("GET", "https://api.nvidia.com/test")
        
        # Check that the request was made with the Bearer token header
        mock_request.assert_called_once()
        call_args = mock_request.call_args
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test-api-key"
        
    @patch('cuopt_sh_client.cuopt_web_hosted_client.requests.request')
    def test_authentication_error_handling(self, mock_request):
        """Test handling of authentication errors."""
        # Test 401 Unauthorized
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_request.return_value = mock_response
        
        client = CuOptServiceWebHostedClient(
            endpoint="https://api.nvidia.com/cuopt/v1",
            api_key="invalid-key"
        )
        
        with pytest.raises(ValueError, match="Authentication failed"):
            client._make_http_request("GET", "https://api.nvidia.com/test")
            
        # Test 403 Forbidden
        mock_response.status_code = 403
        mock_request.return_value = mock_response
        
        with pytest.raises(ValueError, match="Access forbidden"):
            client._make_http_request("GET", "https://api.nvidia.com/test")
            
    def test_base_path_handling(self):
        """Test custom base path handling."""
        client = CuOptServiceWebHostedClient(
            endpoint="https://api.nvidia.com",
            base_path="/custom/path"
        )
        assert client._parsed_endpoint["path"] == "/custom/path"
        assert client.request_url == "https://api.nvidia.com/custom/path"


class TestCreateClientFactory:
    """Test suite for the create_client factory function."""
    
    def test_creates_web_hosted_client_with_endpoint(self):
        """Test that web-hosted client is created when endpoint is provided."""
        client = create_client(endpoint="https://api.nvidia.com/cuopt/v1")
        assert isinstance(client, CuOptServiceWebHostedClient)
        
    def test_creates_web_hosted_client_with_endpoint_and_auth(self):
        """Test that web-hosted client is created with endpoint and auth."""
        client = create_client(
            endpoint="https://api.nvidia.com/cuopt/v1",
            api_key="test-key"
        )
        assert isinstance(client, CuOptServiceWebHostedClient)
        
    def test_error_when_auth_without_endpoint(self):
        """Test that error is raised when auth is provided without endpoint."""
        with pytest.raises(ValueError, match="api_key provided but no endpoint"):
            create_client(api_key="test-key")
        
    def test_creates_self_hosted_client_by_default(self):
        """Test that self-hosted client is created by default."""
        client = create_client(ip="192.168.1.100", port="8080")
        assert isinstance(client, CuOptServiceSelfHostClient)
        assert not isinstance(client, CuOptServiceWebHostedClient)
        
    def test_passes_parameters_correctly(self):
        """Test that parameters are passed correctly to the client."""
        client = create_client(
            endpoint="https://api.nvidia.com/cuopt/v1",
            api_key="test-key",
            polling_timeout=300,
            result_type=mime_type.JSON
        )
        assert isinstance(client, CuOptServiceWebHostedClient)
        assert client.api_key == "test-key"
        assert client.timeout == 300
        assert client.accept_type == mime_type.JSON


class TestCertificateHandling:
    """Test suite for certificate handling."""
        
    def test_self_signed_cert_parameter(self):
        """Test that self_signed_cert parameter is handled correctly."""
        client = CuOptServiceWebHostedClient(
            endpoint="https://api.nvidia.com/cuopt/v1",
            self_signed_cert="/path/to/cert.pem"
        )
        assert client.verify == "/path/to/cert.pem"
        
    def test_https_verification_default(self):
        """Test that HTTPS verification is enabled by default."""
        client = CuOptServiceWebHostedClient(
            endpoint="https://api.nvidia.com/cuopt/v1"
        )
        assert client.verify is True


if __name__ == "__main__":
    pytest.main([__file__])
