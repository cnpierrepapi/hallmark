"""Keep the test suite off the real bucket.

The api module loads the env file at import, so a test run has live credentials
sitting in the environment whether it wants them or not. Any code path that
reaches storage.client() without being stubbed will therefore talk to the
production bucket and succeed, quietly.

That is not hypothetical. Recording attempts from the selection path went in
without the selection fixture stubbing the client, and four suite runs wrote
sixty fabricated attempts into the published ledger before the count on the
homepage gave it away. The rows were fixtures: seventy-two pixel squares
recorded next to real campaign renders.

So the client is refused by default and every test that needs one provides its
own. A test that forgets now fails, or is swallowed by the same best-effort
handling that protects a visitor, rather than reaching the real bucket.
"""

from __future__ import annotations

import pytest

from hallmark import storage


@pytest.fixture(autouse=True)
def no_real_storage(monkeypatch):
    def refuse(*args, **kwargs):
        raise RuntimeError(
            "This test reached hallmark.storage.client(). Tests must never touch "
            "the real bucket: stub the client, as tests/test_ledger.py does."
        )

    monkeypatch.setattr(storage, "client", refuse)
