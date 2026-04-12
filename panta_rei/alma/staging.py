"""Weblog staging: find, extract, and publish ALMA pipeline weblogs.

Ported from the legacy ``stage_weblogs.py`` into the package so it can
be imported regardless of working directory.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import shutil
import sqlite3
import tarfile
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from panta_rei.core.text import now_iso
from panta_rei.core.uid import UID_CORE_RE, canonical_uid, sanitize_uid

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

WEBLOG_TGZ_RE = re.compile(r"\.weblog\.tgz$", re.IGNORECASE)


def extract_uid_from_filename(filename: str) -> Optional[str]:
    """Extract and sanitize a UID from a filename (not full path)."""
    m = UID_CORE_RE.search(filename)
    if m:
        return sanitize_uid(m.group(1))
    return None


def find_mous_directory(archive_path: Path) -> Optional[Path]:
    """Walk up from a weblog archive to find the member.uid___ directory."""
    current = archive_path.parent
    for _ in range(3):
        current = current.parent
        if current.name.startswith("member.uid_"):
            return current
    return None


def path_to_url(path: Path, url_mappings: Dict[str, str]) -> Optional[str]:
    """Convert a filesystem path to a public URL via configured mappings."""
    path_str = str(path.resolve())
    for fs_prefix, url_prefix in url_mappings.items():
        if path_str.startswith(fs_prefix):
            rel_path = path_str[len(fs_prefix):].lstrip("/")
            return f"{url_prefix.rstrip('/')}/{rel_path}"
    return None


def make_world_readable(path: Path) -> None:
    """Recursively set directories to 755 and files to 644."""
    for root, dirs, files in os.walk(path):
        for d in dirs:
            try:
                (Path(root) / d).chmod(0o755)
            except OSError as e:
                log.warning("Could not set permissions on %s: %s", Path(root) / d, e)
        for f in files:
            try:
                (Path(root) / f).chmod(0o644)
            except OSError as e:
                log.warning("Could not set permissions on %s: %s", Path(root) / f, e)
    try:
        path.chmod(0o755)
    except OSError as e:
        log.warning("Could not set permissions on %s: %s", path, e)


def find_weblog_index(weblog_dir: Path) -> Optional[Path]:
    """Find the index.html inside an extracted weblog directory."""
    patterns = [
        weblog_dir / "html" / "index.html",
        weblog_dir / "*/html/index.html",
        weblog_dir / "pipeline*/html/index.html",
    ]
    for pattern in patterns:
        if "*" in str(pattern):
            matches = glob.glob(str(pattern))
            if matches:
                return Path(matches[0])
        elif pattern.exists():
            return pattern

    matches = glob.glob(str(weblog_dir / "**/index.html"), recursive=True)
    for m in matches:
        mp = Path(m)
        if mp.parent.name == "html":
            return mp
    return None


# ---------------------------------------------------------------------------
# WeblogStateDB — thin wrapper around the obs table's weblog columns
# ---------------------------------------------------------------------------

class WeblogStateDB:
    """Track weblog staging status in the obs table."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._ensure_weblog_columns()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _ensure_weblog_columns(self):
        with self._conn() as con:
            existing = {row[1] for row in con.execute("PRAGMA table_info(obs)")}
            for col, typ in [
                ("weblog_staged", "INTEGER DEFAULT 0"),
                ("weblog_path", "TEXT"),
                ("weblog_url", "TEXT"),
                ("weblog_staged_at", "TEXT"),
            ]:
                if col not in existing:
                    try:
                        con.execute(f"ALTER TABLE obs ADD COLUMN {col} {typ}")
                    except sqlite3.OperationalError:
                        pass

    def mark_weblog_staged(
        self, uid: str, weblog_path: Path, weblog_url: Optional[str] = None
    ) -> bool:
        uidc = canonical_uid(uid)
        if not uidc:
            return False
        now = now_iso()
        with self._conn() as con:
            con.execute(
                """UPDATE obs SET weblog_staged=1, weblog_path=?,
                   weblog_url=?, weblog_staged_at=?, updated_at=?
                   WHERE uid=?""",
                (str(weblog_path), weblog_url, now, now, uidc),
            )
            return con.total_changes > 0

    def get_weblog_status(self, uid: str) -> Optional[dict]:
        uidc = canonical_uid(uid)
        if not uidc:
            return None
        with self._conn() as con:
            con.row_factory = sqlite3.Row
            row = con.execute(
                "SELECT weblog_staged, weblog_path, weblog_url FROM obs WHERE uid=?",
                (uidc,),
            ).fetchone()
            return dict(row) if row else None

    def get_unstaged_extracted(self) -> List[dict]:
        with self._conn() as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT uid, status, extracted_root, tar_path FROM obs "
                "WHERE status='extracted' AND (weblog_staged IS NULL OR weblog_staged=0)"
            ).fetchall()
            return [dict(r) for r in rows]

    def reset_weblog_status(self, uid: str) -> bool:
        uidc = canonical_uid(uid)
        if not uidc:
            return False
        now = now_iso()
        with self._conn() as con:
            con.execute(
                """UPDATE obs SET weblog_staged=0, weblog_path=NULL,
                   weblog_url=NULL, weblog_staged_at=NULL, updated_at=?
                   WHERE uid=?""",
                (now, uidc),
            )
            return con.total_changes > 0

    def reset_for_redownload(self, uid: str) -> bool:
        self.reset_weblog_status(uid)
        uidc = canonical_uid(uid)
        if not uidc:
            return False
        now = now_iso()
        with self._conn() as con:
            con.execute(
                """UPDATE obs SET status='pending', tar_path=NULL, tar_deleted=0,
                   extracted_root=NULL, n_extracted=NULL, n_skipped=NULL, updated_at=?
                   WHERE uid=?""",
                (now, uidc),
            )
            return con.total_changes > 0


# ---------------------------------------------------------------------------
# WeblogStager
# ---------------------------------------------------------------------------

class WeblogStager:
    """Find and stage weblogs from extracted ALMA data."""

    def __init__(
        self,
        base_dir: Path,
        weblog_dir: Path,
        db: Optional[WeblogStateDB] = None,
        url_mappings: Optional[Dict[str, str]] = None,
        dry_run: bool = False,
    ):
        self.base_dir = Path(base_dir).resolve()
        self.weblog_dir = Path(weblog_dir).resolve()
        self.url_mappings = url_mappings or {
            "/scratch/almanas": "https://www.alma.ac.uk/nas",
        }
        self.dry_run = dry_run

        if db is None:
            db_path = self.base_dir / "alma_retrieval_state.sqlite3"
            if db_path.exists():
                self.db = WeblogStateDB(db_path)
            else:
                log.warning("State database not found: %s", db_path)
                self.db = None
        else:
            self.db = db

        self.corrupted_count = 0

    def find_weblog_archives(self) -> List[Tuple[str, Path]]:
        archives = []
        pattern = str(self.base_dir / "**" / "*.weblog.tgz")
        for match in glob.glob(pattern, recursive=True):
            archive_path = Path(match)
            uid = extract_uid_from_filename(archive_path.name)
            if uid:
                archives.append((uid, archive_path))
            else:
                log.warning("Could not extract UID from filename: %s", archive_path.name)
        log.info("Found %d weblog archives", len(archives))
        return archives

    def is_already_staged(self, uid: str) -> bool:
        if self.db is None:
            target_dir = self.weblog_dir / sanitize_uid(uid)
            if target_dir.exists():
                return find_weblog_index(target_dir) is not None
            return False
        status = self.db.get_weblog_status(uid)
        if status and status.get("weblog_staged"):
            weblog_path = status.get("weblog_path")
            if weblog_path and Path(weblog_path).exists():
                return True
        return False

    def handle_corrupted_mous(self, uid: str, archive_path: Path) -> None:
        log.warning("Handling corrupted MOUS: %s", uid)
        mous_dir = find_mous_directory(archive_path)
        if mous_dir and mous_dir.exists():
            log.info("Deleting corrupted MOUS directory: %s", mous_dir)
            if not self.dry_run:
                try:
                    shutil.rmtree(mous_dir)
                except OSError as e:
                    log.error("Failed to delete MOUS directory %s: %s", mous_dir, e)
                    return
        uid_sanitized = sanitize_uid(uid)
        staged_dir = self.weblog_dir / uid_sanitized
        if staged_dir.exists():
            if not self.dry_run:
                try:
                    shutil.rmtree(staged_dir)
                except OSError as e:
                    log.error("Failed to delete staged directory %s: %s", staged_dir, e)
        if self.db and not self.dry_run:
            if self.db.reset_for_redownload(uid):
                log.info("Reset %s to pending status for re-download", uid)
                self.corrupted_count += 1

    def extract_weblog(self, uid: str, archive_path: Path) -> Optional[Path]:
        uid_sanitized = sanitize_uid(uid)
        target_dir = self.weblog_dir / uid_sanitized
        if self.dry_run:
            log.info("[DRY-RUN] Would extract %s -> %s", archive_path.name, target_dir)
            return target_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(archive_path, "r:gz") as tf:
                for member in tf.getmembers():
                    member_path = target_dir / member.name
                    try:
                        member_path.resolve().relative_to(target_dir.resolve())
                    except ValueError:
                        log.warning("Skipping unsafe path: %s", member.name)
                        continue
                    if member.isdir():
                        member_path.mkdir(parents=True, exist_ok=True)
                    else:
                        member_path.parent.mkdir(parents=True, exist_ok=True)
                        tf.extract(member, path=target_dir)
            make_world_readable(target_dir)
            index_path = find_weblog_index(target_dir)
            if index_path:
                return index_path.parent.parent
            return target_dir
        except tarfile.TarError as e:
            if any(k in str(e).lower() for k in ["zlib", "decompressing", "error -3"]):
                log.error("Corrupted archive detected: %s: %s", archive_path, e)
                self.handle_corrupted_mous(uid, archive_path)
            else:
                log.error("Tar error extracting %s: %s", archive_path, e)
            return None
        except (zlib.error, OSError) as e:
            if any(k in str(e).lower() for k in ["zlib", "decompressing", "error -3"]):
                log.error("Corrupted archive detected: %s: %s", archive_path, e)
                self.handle_corrupted_mous(uid, archive_path)
            else:
                log.error("Error extracting %s: %s", archive_path, e)
            return None

    def stage_weblog(self, uid: str, archive_path: Path) -> Optional[Tuple[Path, str]]:
        if self.is_already_staged(uid):
            return None
        weblog_path = self.extract_weblog(uid, archive_path)
        if weblog_path is None:
            return None
        weblog_url = path_to_url(weblog_path, self.url_mappings)
        if self.db and not self.dry_run:
            self.db.mark_weblog_staged(uid, weblog_path, weblog_url)
        return (weblog_path, weblog_url)

    def stage_all(self) -> List[Tuple[str, Path, Optional[str]]]:
        staged = []
        archives = self.find_weblog_archives()
        by_uid: Dict[str, List[Path]] = {}
        for uid, archive_path in archives:
            canonical = canonical_uid(uid)
            if canonical:
                by_uid.setdefault(canonical, []).append(archive_path)

        log.info("Processing %d unique UIDs", len(by_uid))
        for uid, archive_paths in sorted(by_uid.items()):
            archive_path = archive_paths[0]
            result = self.stage_weblog(uid, archive_path)
            if result:
                weblog_path, weblog_url = result
                staged.append((uid, weblog_path, weblog_url))
                log.info("Staged: %s", uid)
                if weblog_url:
                    log.info("  URL: %s", weblog_url)

        log.info("Staged %d weblogs", len(staged))
        return staged

    def summary(self) -> str:
        if self.db is None:
            return "No database available"
        with self.db._conn() as con:
            total = con.execute("SELECT COUNT(*) FROM obs").fetchone()[0]
            extracted = con.execute(
                "SELECT COUNT(*) FROM obs WHERE status='extracted'"
            ).fetchone()[0]
            staged_count = con.execute(
                "SELECT COUNT(*) FROM obs WHERE weblog_staged=1"
            ).fetchone()[0]
            pending = con.execute(
                "SELECT COUNT(*) FROM obs WHERE status='pending'"
            ).fetchone()[0]
        s = f"total={total} | extracted={extracted} | weblogs_staged={staged_count}"
        if self.corrupted_count > 0:
            s += f" | corrupted_reset={self.corrupted_count}"
        if pending > 0:
            s += f" | pending_redownload={pending}"
        return s
