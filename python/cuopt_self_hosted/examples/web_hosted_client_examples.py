#!/usr/bin/env python3

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

"""
Examples demonstrating the usage of CuOptServiceWebHostedClient.

This module provides comprehensive examples of how to use the web-hosted
client for various scenarios including different authentication methods,
endpoint configurations, and problem types.
"""

import os
import json
from cuopt_sh_client import CuOptServiceWebHostedClient, create_client


def example_basic_web_hosted_client():
    """Basic example using web-hosted client with API key."""
    print("=== Basic Web-Hosted Client Example ===")
    
    # Create client with endpoint and API key
    client = CuOptServiceWebHostedClient(
        endpoint="https://api.nvidia.com/cuopt/v1",
        api_key="your-api-key-here"  # Or set CUOPT_API_KEY environment variable
    )
    
    # Example problem data (simplified)
    problem_data = {
        "cost_matrix": [[0, 10, 15], [10, 0, 20], [15, 20, 0]],
        "fleet_data": {"vehicle_count": 1},
        "task_data": {"demand": [0, 1, 1]},
        "solver_config": {"time_limit": 10}
    }
    
    try:
        # Solve the problem
        result = client.get_optimized_routes(problem_data)
        print(f"Solution found: {result}")
    except Exception as e:
        print(f"Error: {e}")


def example_bearer_token_authentication():
    """Example using bearer token authentication."""
    print("\n=== Bearer Token Authentication Example ===")
    
    client = CuOptServiceWebHostedClient(
        endpoint="https://inference.nvidia.com/cuopt",
        bearer_token="your-bearer-token-here"  # Or set CUOPT_BEARER_TOKEN env var
    )
    
    # Rest of the example would be similar to basic example
    print("Client created with bearer token authentication")


def example_environment_variables():
    """Example using environment variables for authentication."""
    print("\n=== Environment Variables Example ===")
    
    # Set environment variables (in practice, these would be set externally)
    os.environ["CUOPT_API_KEY"] = "your-api-key-from-env"
    
    # Client will automatically pick up the API key from environment
    client = CuOptServiceWebHostedClient(
        endpoint="https://api.nvidia.com/cuopt/v1"
    )
    
    print("Client created using API key from environment variable")


def example_custom_endpoint_with_path():
    """Example with custom endpoint including path."""
    print("\n=== Custom Endpoint with Path Example ===")
    
    client = CuOptServiceWebHostedClient(
        endpoint="https://my-custom-service.com:8080/api/v2/cuopt",
        api_key="custom-service-key"
    )
    
    print(f"Request URL: {client.request_url}")
    print(f"Solution URL: {client.solution_url}")


def example_factory_function():
    """Example using the factory function to create clients."""
    print("\n=== Factory Function Examples ===")
    
    # Creates web-hosted client
    web_client = create_client(
        endpoint="https://api.nvidia.com/cuopt/v1",
        api_key="your-key"
    )
    print(f"Created: {type(web_client).__name__}")
    
    # Creates self-hosted client (legacy mode)
    self_hosted_client = create_client(
        ip="192.168.1.100",
        port="5000"
    )
    print(f"Created: {type(self_hosted_client).__name__}")


def example_linear_programming():
    """Example solving a linear programming problem."""
    print("\n=== Linear Programming Example ===")
    
    client = CuOptServiceWebHostedClient(
        endpoint="https://api.nvidia.com/cuopt/v1",
        api_key="your-api-key-here"
    )
    
    # Example MPS file path or problem data
    mps_file_path = "example_problem.mps"  # This would be your actual MPS file
    
    try:
        # Solve LP problem
        result = client.get_LP_solve(
            mps_file_path,
            response_type="obj"  # Returns ThinClientSolution object
        )
        print(f"LP solution status: {result['status']}")
        print(f"Objective value: {result['solution'].primal_objective}")
    except Exception as e:
        print(f"Error solving LP: {e}")


def example_with_callbacks():
    """Example using callbacks for MIP solver logs and incumbents."""
    print("\n=== Callbacks Example ===")
    
    client = CuOptServiceWebHostedClient(
        endpoint="https://api.nvidia.com/cuopt/v1",
        api_key="your-api-key-here"
    )
    
    def logging_callback(log_lines):
        """Callback to handle solver logs."""
        for line in log_lines:
            print(f"SOLVER LOG: {line}")
    
    def incumbent_callback(solution, cost):
        """Callback to handle incumbent solutions."""
        print(f"New incumbent found with cost: {cost}")
    
    # Example MILP problem data
    problem_data = "example_milp.mps"
    
    try:
        result = client.get_LP_solve(
            problem_data,
            logging_callback=logging_callback,
            incumbent_callback=incumbent_callback,
            response_type="obj"
        )
        print(f"Final solution cost: {result['solution'].primal_objective}")
    except Exception as e:
        print(f"Error: {e}")


def example_error_handling():
    """Example demonstrating error handling."""
    print("\n=== Error Handling Example ===")
    
    # Example with invalid API key
    client = CuOptServiceWebHostedClient(
        endpoint="https://api.nvidia.com/cuopt/v1",
        api_key="invalid-key"
    )
    
    try:
        result = client.get_optimized_routes({"invalid": "data"})
    except ValueError as e:
        if "Authentication failed" in str(e):
            print("Authentication error - check your API key")
        elif "Access forbidden" in str(e):
            print("Access forbidden - check your permissions")
        else:
            print(f"Other error: {e}")


def example_timeout_and_polling():
    """Example demonstrating timeout and polling behavior."""
    print("\n=== Timeout and Polling Example ===")
    
    client = CuOptServiceWebHostedClient(
        endpoint="https://api.nvidia.com/cuopt/v1",
        api_key="your-api-key-here",
        polling_timeout=60,  # Timeout after 60 seconds
        timeout_exception=False  # Return request ID instead of raising exception
    )
    
    problem_data = {"large": "problem_data"}
    
    try:
        result = client.get_optimized_routes(problem_data)
        
        if isinstance(result, dict) and "reqId" in result:
            print(f"Request timed out, got request ID: {result['reqId']}")
            
            # Later, you can poll for the result
            final_result = client.repoll(result["reqId"])
            print(f"Final result: {final_result}")
        else:
            print(f"Solution completed: {result}")
            
    except Exception as e:
        print(f"Error: {e}")


def example_backward_compatibility():
    """Example showing backward compatibility with legacy parameters."""
    print("\n=== Backward Compatibility Example ===")
    
    # This still works (legacy mode)
    legacy_client = CuOptServiceWebHostedClient(
        ip="192.168.1.100",
        port="5000",
        use_https=True
    )
    print(f"Legacy client URL: {legacy_client.request_url}")
    
    # But if you provide endpoint, it takes precedence
    mixed_client = CuOptServiceWebHostedClient(
        endpoint="https://api.nvidia.com/cuopt/v1",
        ip="192.168.1.100",  # This will be ignored
        port="5000",         # This will be ignored
        api_key="your-key"
    )
    print(f"Mixed client URL: {mixed_client.request_url}")


def main():
    """Run all examples."""
    print("CuOpt Web-Hosted Client Examples")
    print("=" * 50)
    
    # Note: These examples won't actually run without valid credentials
    # and endpoints. They're provided for demonstration purposes.
    
    example_basic_web_hosted_client()
    example_bearer_token_authentication()
    example_environment_variables()
    example_custom_endpoint_with_path()
    example_factory_function()
    example_linear_programming()
    example_with_callbacks()
    example_error_handling()
    example_timeout_and_polling()
    example_backward_compatibility()
    
    print("\n" + "=" * 50)
    print("Examples completed. Note: Actual execution requires valid")
    print("credentials and endpoints.")


if __name__ == "__main__":
    main()
