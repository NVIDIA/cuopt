# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Legacy-only cuOpt HTTP server modules. Permanent utils must not import
# this package (see tests/test_utils_deprecated_boundary.py). Deleting the
# old server is: remove cuopt_service.py, webserver.py, this package, and
# tests that exist only for the local job queue.
