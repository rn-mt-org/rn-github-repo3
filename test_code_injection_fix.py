"""
Tests for CWE-94 Code Injection fix in python-vul-file.py

The vulnerability was an exec() call that allowed arbitrary Python code execution
via the `include` URL parameter. The fix replaces exec() with safe file content
reading, breaking the taint flow from user-controlled input to code execution.
"""

import http.client
import io
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request

# We import the module-under-test by loading it directly via importlib
# so we can start a test server without polluting the global environment.
import importlib.util
import types

# ---------------------------------------------------------------------------
# Helpers: start the server in a background thread and tear it down after.
# ---------------------------------------------------------------------------

# Use a high ephemeral port to avoid conflicts.
TEST_PORT = 65499


def _load_module():
    """Load python-vul-file as a module without executing its __main__ block."""
    spec = importlib.util.spec_from_file_location(
        "dsvw",
        os.path.join(os.path.dirname(__file__), "python-vul-file.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # Patch LISTEN_PORT before executing the module body so the server binds
    # on our test port.  We cannot inject after the fact because the constant
    # is evaluated at import time.
    spec.loader.exec_module(mod)
    return mod


class CodeInjectionFixTest(unittest.TestCase):
    """Verify that the exec() code-injection sink has been removed."""

    # ------------------------------------------------------------------
    # Class-level: spin up the embedded HTTP server once for all tests.
    # ------------------------------------------------------------------
    _server = None
    _server_thread = None
    _mod = None

    @classmethod
    def setUpClass(cls):
        cls._mod = _load_module()
        # Override the listen port so we don't conflict with a running instance.
        cls._mod.LISTEN_PORT = TEST_PORT
        cls._mod.LISTEN_ADDRESS = "127.0.0.1"
        cls._mod.init()
        import http.server
        import socketserver

        class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        cls._server = ThreadingServer(
            ("127.0.0.1", TEST_PORT), cls._mod.ReqHandler
        )
        cls._server_thread = threading.Thread(
            target=cls._server.serve_forever, daemon=True
        )
        cls._server_thread.start()

    @classmethod
    def tearDownClass(cls):
        if cls._server:
            cls._server.shutdown()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _get(self, path_and_query: str) -> tuple:
        """Make a GET request to the test server. Returns (status, body)."""
        url = "http://127.0.0.1:%d%s" % (TEST_PORT, path_and_query)
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                return resp.status, resp.read().decode(errors="replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode(errors="replace")

    # ------------------------------------------------------------------
    # Core security tests: exec() must NOT be called
    # ------------------------------------------------------------------

    def test_include_with_local_file_does_not_execute_code(self):
        """
        A Python script written to a temp file must NOT be executed by the
        server.  Before the fix, exec() would run it; after the fix, the
        content is only returned as plain text.
        """
        # Write a Python script that appends a sentinel string to a temp file
        # if it is ever executed.
        sentinel_path = os.path.join(tempfile.gettempdir(), "dsvw_exec_sentinel.txt")
        # Remove any leftover sentinel from a previous run.
        if os.path.exists(sentinel_path):
            os.remove(sentinel_path)

        script_content = (
            "import builtins\n"
            "open(%r, 'w').write('EXECUTED')\n" % sentinel_path
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(script_content)
            tmp_path = tmp.name

        try:
            status, body = self._get("/?include=" + urllib.parse.quote(tmp_path))
            # The server may return 200 or 500 depending on whether reading
            # succeeds; what must NOT happen is code execution.
            self.assertFalse(
                os.path.exists(sentinel_path),
                "exec() was invoked: the sentinel file was created, meaning "
                "arbitrary code from 'include' was executed (CWE-94 still present).",
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if os.path.exists(sentinel_path):
                os.remove(sentinel_path)

    def test_include_returns_file_contents_as_text(self):
        """
        After the fix, the `include` parameter should return the raw file
        content as text (not execute it).
        """
        marker = "SAFE_CONTENT_MARKER_12345"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as tmp:
            tmp.write(marker)
            tmp_path = tmp.name

        try:
            status, body = self._get("/?include=" + urllib.parse.quote(tmp_path))
            # The raw content should appear somewhere in the response body.
            self.assertIn(
                marker,
                body,
                "Expected file content to appear in response body after the fix.",
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_exec_not_called_with_python_code_payload(self):
        """
        Craft a payload that would only have an effect if exec() is called.
        Write a script that sets a module-level flag; verify the flag is
        not set after the request.
        """
        # Write a Python script that would set a global attribute on a temp
        # module if executed.
        script_content = "import builtins; builtins.__DSVW_INJECTED__ = True\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(script_content)
            tmp_path = tmp.name

        try:
            import builtins
            # Ensure the sentinel attribute doesn't exist before the test.
            if hasattr(builtins, "__DSVW_INJECTED__"):
                delattr(builtins, "__DSVW_INJECTED__")

            self._get("/?include=" + urllib.parse.quote(tmp_path))

            self.assertFalse(
                hasattr(builtins, "__DSVW_INJECTED__"),
                "exec() was invoked: builtins.__DSVW_INJECTED__ was set, "
                "meaning arbitrary Python code was executed (CWE-94 not fixed).",
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            # Clean up in case the test failed and the attribute was set.
            import builtins
            if hasattr(builtins, "__DSVW_INJECTED__"):
                delattr(builtins, "__DSVW_INJECTED__")

    # ------------------------------------------------------------------
    # Regression / functionality tests
    # ------------------------------------------------------------------

    def test_normal_homepage_still_works(self):
        """The root path without parameters should return HTTP 200."""
        status, body = self._get("/")
        self.assertEqual(200, status, "Homepage should return HTTP 200.")
        self.assertIn("<!DOCTYPE html>", body, "Homepage should return HTML.")

    def test_user_query_still_works(self):
        """User ID lookup should still work (no regression)."""
        status, body = self._get("/?id=1")
        self.assertEqual(200, status)
        self.assertIn("admin", body)

    def test_404_for_unknown_path(self):
        """Unknown paths should still return 404."""
        status, _body = self._get("/nonexistent-path")
        self.assertEqual(404, status)

    def test_include_missing_file_returns_error(self):
        """
        Requesting a non-existent file via `include` should result in an
        error response (500), not silent code execution.
        """
        status, _body = self._get("/?include=/nonexistent/path/to/file.py")
        self.assertEqual(
            500,
            status,
            "A missing include file should produce a 500, not silently succeed.",
        )

    # ------------------------------------------------------------------
    # Source-code / static analysis verification
    # ------------------------------------------------------------------

    def test_exec_is_not_in_include_branch_source(self):
        """
        Verify statically that exec() does not appear in the handler source
        after the fix.  This acts as a canary: if someone re-introduces the
        exec() call, this test will fail immediately.
        """
        src_path = os.path.join(os.path.dirname(__file__), "python-vul-file.py")
        with open(src_path, "r") as fh:
            source = fh.read()

        # Locate the `include` branch in the source.
        include_idx = source.find('"include" in params')
        self.assertGreater(include_idx, 0, "Could not locate 'include' branch in source.")

        # Locate the *next* `elif` or `else` after the include branch to
        # isolate just the include block.
        next_elif_idx = source.find("elif", include_idx + 1)
        include_block = source[include_idx:next_elif_idx] if next_elif_idx > 0 else source[include_idx:]

        self.assertNotIn(
            "exec(",
            include_block,
            "exec() was found in the 'include' branch — the CWE-94 code "
            "injection vulnerability has NOT been fixed.",
        )


if __name__ == "__main__":
    unittest.main()
