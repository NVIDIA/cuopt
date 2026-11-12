# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import errno
import http.client
import io
import json
import os
from pathlib import Path
import runpy
import socket
import ssl
import unittest
from unittest import mock
import urllib.error


SCRIPT = Path(__file__).resolve().parents[1] / "utils" / "pr_test_summary.py"
ENV = {
    "GH_TOKEN": "test-token",
    "GITHUB_REPOSITORY": "owner/repo",
    "GITHUB_RUN_ID": "123",
    "GITHUB_REF": "refs/heads/pull-request/42",
}


def http_error(code, headers=None):
    error = urllib.error.HTTPError(
        "https://api.github.com/test", code, "test error", headers or {}, None
    )
    error.close()
    return error


class Response:
    def __init__(self, value):
        self.data = json.dumps(value).encode()
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.data


class SummaryEntrypointTest(unittest.TestCase):
    def execute(self, urlopen, env=ENV):
        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch("urllib.request.urlopen", side_effect=urlopen) as call,
            mock.patch("time.sleep"),
            contextlib.redirect_stderr(output),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            runpy.run_path(str(SCRIPT), run_name="__main__")
        return output.getvalue(), call

    def test_persistent_transient_errors_are_warnings(self):
        errors = [
            http_error(502),
            http_error(429),
            http_error(403, {"Retry-After": "1"}),
            urllib.error.URLError("temporary DNS error"),
            TimeoutError("timed out"),
            ConnectionResetError("reset"),
            OSError(errno.ENETUNREACH, "network unreachable"),
            socket.gaierror("lookup failed"),
            ssl.SSLError("TLS handshake failed"),
            http.client.RemoteDisconnected("connection closed"),
        ]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                output, calls = self.execute(error)
                self.assertEqual(calls.call_count, 4)
                self.assertIn("Warning:", output)
                self.assertNotIn(ENV["GH_TOKEN"], output)

    def test_rate_limit_with_long_retry_after_warns_without_waiting(self):
        output, calls = self.execute(http_error(429, {"Retry-After": "3600"}))
        self.assertEqual(calls.call_count, 1)
        self.assertIn("Warning:", output)

    def test_nontransient_http_and_program_errors_still_raise(self):
        for error in (
            http_error(401),
            http_error(403),
            http_error(404),
            ssl.SSLCertVerificationError(1, "bad certificate"),
            urllib.error.URLError(
                ssl.SSLCertVerificationError(1, "bad certificate")
            ),
            FileNotFoundError("missing local file"),
            ValueError("bad data"),
        ):
            with self.subTest(error=repr(error)):
                with self.assertRaises(type(error)):
                    self.execute(error)

    def test_missing_environment_and_bad_json_still_raise(self):
        with self.assertRaises(KeyError):
            self.execute(None, env={})

        class InvalidJSON(Response):
            def read(self):
                return b"invalid JSON"

        with self.assertRaises(json.JSONDecodeError):
            self.execute(lambda *args, **kwargs: InvalidJSON(None))

    def test_successful_summary_is_still_posted(self):
        requests = []

        def urlopen(request, **kwargs):
            requests.append(request)
            if request.full_url.endswith("/jobs?per_page=100"):
                return Response(
                    {
                        "jobs": [
                            {
                                "name": "conda-python-tests",
                                "conclusion": "success",
                            }
                        ]
                    }
                )
            if request.get_method() == "GET":
                return Response([])
            return Response({"id": 7})

        _, calls = self.execute(urlopen)
        self.assertEqual(calls.call_count, 3)
        self.assertEqual(requests[-1].get_method(), "POST")
        body = json.loads(requests[-1].data)["body"]
        self.assertIn("All 1 test job(s) passed", body)

    def test_job_log_transport_error_is_best_effort(self):
        def urlopen(request, **kwargs):
            return Response(
                {
                    "jobs": [
                        {
                            "name": "conda-python-tests",
                            "conclusion": "failure",
                            "id": 17,
                            "html_url": "https://example.test/job/17",
                        }
                    ]
                }
            )

        with mock.patch("urllib.request.build_opener") as opener:
            opener.return_value.open.side_effect = TimeoutError(
                "log timed out"
            )
            output, calls = self.execute(urlopen)
        self.assertEqual(calls.call_count, 1)
        self.assertIn("Warning:", output)

    def test_local_stdout_error_does_not_retry_or_get_swallowed(self):
        requests = []

        def urlopen(request, **kwargs):
            requests.append(request)
            if request.full_url.endswith("/jobs?per_page=100"):
                return Response(
                    {
                        "jobs": [
                            {
                                "name": "conda-python-tests",
                                "conclusion": "success",
                            }
                        ]
                    }
                )
            if request.get_method() == "GET":
                return Response([])
            return Response({"id": 7})

        with mock.patch(
            "builtins.print", side_effect=BrokenPipeError("stdout closed")
        ):
            with self.assertRaises(BrokenPipeError):
                self.execute(urlopen)
        self.assertEqual(len(requests), 3)

    def test_persistent_post_error_is_best_effort_after_rechecking(self):
        posts = []
        comment_lookups = []

        def urlopen(request, **kwargs):
            if request.full_url.endswith("/jobs?per_page=100"):
                return Response(
                    {
                        "jobs": [
                            {
                                "name": "conda-python-tests",
                                "conclusion": "success",
                            }
                        ]
                    }
                )
            if request.get_method() == "GET":
                comment_lookups.append(request)
                return Response([])
            posts.append(request)
            raise http_error(502)

        output, _ = self.execute(urlopen)
        self.assertEqual(len(posts), 4)
        self.assertEqual(len(comment_lookups), 4)
        self.assertIn("Warning:", output)


if __name__ == "__main__":
    unittest.main()
