"""ALMA query helpers and server fallback.

Extracted from get_data.py — configures astroquery caching, sets up
the Alma client, runs project queries, and filters invalid release dates.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from panta_rei.core.text import as_text

log = logging.getLogger(__name__)

TIMEOUT = 300


def configure_astroquery_cache(cache_root: Path) -> None:
    """Set XDG_CACHE_HOME and ASTROPY_CACHE_DIR so astroquery caches under *cache_root*."""
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    os.environ["ASTROPY_CACHE_DIR"] = str(cache_root / "astropy")
    log.info(f"Astropy/astroquery cache root set to {os.environ['XDG_CACHE_HOME']}")


def setup_alma_client(server_url: str) -> Alma:
    """Return an :class:`Alma` client pointed at *server_url*."""
    from astroquery.alma import Alma

    alma = Alma()
    alma.TIMEOUT = TIMEOUT
    alma.archive_url = server_url
    alma.dataarchive_url = server_url
    cache_dir = os.environ.get("ASTROPY_CACHE_DIR")
    if cache_dir and hasattr(alma, "cache_location"):
        try:
            alma.cache_location = str(Path(cache_dir) / "astroquery" / "Alma")
        except Exception:
            pass
    return alma


def query_project(alma: Alma, project_code: str):
    """Query the ALMA archive for *project_code* and return the results table.

    Automatically filters rows with invalid or far-future release dates.
    """
    log.info(f"Querying project {project_code}")
    results = alma.query(payload=dict(project_code=project_code), public=None)
    log.info(f"Found {len(results)} total results (pre-filter)")
    results = filter_valid_release_dates(results)
    log.info(f"{len(results)} results after date filter")
    return results


def filter_valid_release_dates(results):
    """Remove rows whose ``obs_release_date`` is empty, non-numeric, or starts with ``3000``."""
    try:
        release_dates = results["obs_release_date"]
    except Exception:
        return results
    keep = []
    for d in release_dates:
        s = as_text(d).strip()
        keep.append(bool(s) and s[0].isdigit() and not s.startswith("3000"))
    mask = np.array(keep, dtype=bool)
    dropped = int((~mask).sum())
    if dropped:
        log.info(f"Filtered {dropped} rows with invalid/future release dates")
    return results[mask]
