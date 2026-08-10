"""Fixed, credential-free environment for every local helper process."""

from __future__ import annotations


def sanitized_worker_environment() -> dict[str, str]:
    """Return the complete minimal environment inherited by local workers.

    Workers are launched with an explicit environment instead of filtering the
    ambient process environment.  A denylist can never enumerate every alias a
    caller might use for credentials.
    """

    return {
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
        "TZ": "UTC",
    }
