"""ALMA and GitHub authentication.

Credential resolution priority: CLI arg > env var > systemd credential.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from panta_rei.core.errors import AuthError

# Systemd credential file names
ALMA_CREDENTIAL_NAME = "svc.alma"
GITHUB_CREDENTIAL_NAME = "github.token"

# ALMA authentication URLs
AUTH_URLS = (
    "almascience.nrao.edu",
    "almascience.eso.org",
    "asa.alma.cl",
    "rh-cas.alma.cl",
)


def read_systemd_credential(name: str) -> Optional[str]:
    """Read a credential from the systemd CREDENTIALS_DIRECTORY."""
    cdir = os.environ.get("CREDENTIALS_DIRECTORY")
    if not cdir:
        return None
    p = Path(cdir) / name
    return p.read_text(encoding="utf-8").strip() if p.is_file() else None


def resolve_alma_creds(
    username_cli: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Resolve ALMA username and password.

    Priority: CLI arg > env var > systemd credential.
    """
    username = username_cli or os.environ.get("ALMA_USERNAME")
    password = (
        read_systemd_credential(ALMA_CREDENTIAL_NAME)
        or os.environ.get("ALMA_PASSWORD")
    )
    return username, password


def resolve_github_token() -> Optional[str]:
    """Resolve GitHub token from systemd credential or environment variable."""
    return (
        read_systemd_credential(GITHUB_CREDENTIAL_NAME)
        or os.environ.get("GITHUB_TOKEN")
    )


def install_headless_password(password: str) -> None:
    """Patch astroquery to use a fixed password instead of prompting.

    Must be called before any Alma.login() call in non-interactive mode.
    """
    from astroquery.query import QueryWithLogin

    def _no_prompt_get_password(self, service_name, username, reenter=False):
        return password, password

    QueryWithLogin._get_password = _no_prompt_get_password


def login_alma(alma, username: Optional[str]) -> None:
    """Log in to ALMA archive, handling API variations."""
    try:
        alma.login(username, store_password=False, auth_urls=list(AUTH_URLS))
    except TypeError:
        alma.login(username, store_password=False)
