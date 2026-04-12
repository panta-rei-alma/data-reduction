"""ALMA data retrieval and extraction orchestration.

Extracted from get_data.py — downloads UIDs via astroquery, extracts
tarballs, and updates the obs table through :mod:`panta_rei.db.models`.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np
import requests

if TYPE_CHECKING:
    from astroquery.alma import Alma

from panta_rei.alma.client import (
    configure_astroquery_cache,
    query_project,
    setup_alma_client,
)
from panta_rei.auth import login_alma
from panta_rei.core.tar import safe_extract_tar
from panta_rei.core.text import as_text
from panta_rei.core.uid import canonical_uid, extract_uid_from_path
from panta_rei.db.models import ObsQueries, ObsStatus

log = logging.getLogger(__name__)

# ALMA mirror URLs — fallback order
ALMA_SERVERS = [
    "https://almascience.nrao.edu",
    "https://almascience.eso.org",
    "https://almascience.nao.ac.jp",
]


def retrieve_uids(alma: Alma, uids: List[str]) -> List[Path]:
    """Download *uids* via ``alma.retrieve_data_from_uid`` and return resolved Paths."""
    if not uids:
        return []
    log.info(f"Requesting download for {len(uids)} UIDs")
    files = alma.retrieve_data_from_uid(uids)
    paths = [Path(f).resolve() for f in np.atleast_1d(files)]
    if paths:
        try:
            common = os.path.commonpath([str(p) for p in paths])
            log.info(f"Downloaded {len(paths)} files under {common}")
        except ValueError:
            log.info(f"Downloaded {len(paths)} files (multiple roots)")
    return paths


def extract_single_tar(tar_path: Path, base_dir: Path) -> Tuple[int, int, bool]:
    """Extract one tarball into *base_dir*.

    Returns ``(extracted_count, skipped_count, ok)``.
    """
    log.info(f"Extracting {tar_path.name} -> {base_dir}")
    try:
        with tarfile.open(tar_path, "r") as tf:
            extracted, skipped = safe_extract_tar(tf, base_dir)
        log.info(f"Completed {tar_path.name}: extracted {extracted}, skipped {skipped}")
        return extracted, skipped, True
    except tarfile.TarError as e:
        log.error(f"Error extracting {tar_path}: {e}")
        return 0, 0, False


def retrieve_and_extract(
    username: str,
    project_code: str,
    base_dir: Path,
    db_manager,
) -> bool:
    """Main orchestration: query ALMA, download new UIDs, extract, and update the DB.

    Parameters
    ----------
    username : str
        ALMA username for authentication.
    project_code : str
        ALMA project code (e.g. ``2025.1.00383.L``).
    base_dir : Path
        Root directory for extracted data.
    db_manager : DatabaseManager
        A :class:`~panta_rei.db.connection.DatabaseManager` instance whose
        ``.connect()`` method yields a :class:`sqlite3.Connection`.

    Returns
    -------
    bool
        True if retrieval succeeded (or nothing to do), False if all mirrors failed.

    Raises
    ------
    ALMAError
        If all ALMA mirrors failed and no data could be retrieved.
    """
    cache_root = base_dir / "tars"
    base_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    configure_astroquery_cache(cache_root)

    for server_url in ALMA_SERVERS:
        log.info(f"Connecting to {server_url}")
        try:
            alma = setup_alma_client(server_url)
            login_alma(alma, username)

            results = query_project(alma, project_code)
            if results is None or len(results) == 0:
                log.info(f"No results for {project_code}")
                return True

            raw_uids = results["member_ous_uid"]
            obsids: List[str] = [as_text(u) for u in np.unique(raw_uids).tolist()]

            rel_map: Dict[str, str] = {}
            for uid, rel in zip(results["member_ous_uid"], results["obs_release_date"]):
                key = canonical_uid(uid)
                if key:
                    rel_map[key] = as_text(rel)

            with db_manager.connect() as con:
                for uid in obsids:
                    ObsQueries.upsert_seen(con, uid, rel_map.get(canonical_uid(uid)))

            with db_manager.connect() as con:
                to_download = ObsQueries.uids_to_download(con, obsids)
            log.info(f"Archive has {len(obsids)} UIDs; queued this run: {len(to_download)}")
            if not to_download:
                log.info("Nothing to do; all UIDs already extracted.")
                return True

            paths = retrieve_uids(alma, to_download)

            uid_path_pairs = [(extract_uid_from_path(p.name), p) for p in paths]
            with db_manager.connect() as con:
                ObsQueries.mark_many_downloaded(
                    con, [(u, p) for u, p in uid_path_pairs if u]
                )

            for tar_path in paths:
                if tar_path.suffix.lower() != ".tar":
                    continue
                uidc = extract_uid_from_path(tar_path.name) or "unknown"
                extracted, skipped, ok = extract_single_tar(tar_path, base_dir)
                if ok:
                    tar_deleted = False
                    try:
                        sz_gb = tar_path.stat().st_size / (1024 ** 3)
                        tar_path.unlink()
                        tar_deleted = True
                        log.info(f"Deleted {tar_path.name} (freed {sz_gb:.2f} GB)")
                    except OSError as e:
                        log.error(f"Failed to delete {tar_path}: {e}")
                    if uidc != "unknown":
                        with db_manager.connect() as con:
                            ObsQueries.mark_extracted(
                                con, uidc, base_dir, extracted, skipped, tar_deleted
                            )
                else:
                    if uidc != "unknown":
                        with db_manager.connect() as con:
                            ObsQueries.mark_error(con, uidc)
            return True

        except requests.exceptions.ReadTimeout as e:
            log.warning(f"Timeout on {server_url}: {e}")
            continue
        except requests.exceptions.HTTPError as e:
            log.warning(f"HTTP error on {server_url}: {e}")
            continue
        except Exception as e:
            log.error(f"Unexpected error on {server_url}: {e}", exc_info=True)
            continue

    from panta_rei.core.errors import ALMAError

    raise ALMAError("Failed on all ALMA mirrors")
