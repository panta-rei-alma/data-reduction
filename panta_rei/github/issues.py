"""GitHub issue management for ALMA scheduling blocks.

Creates and updates one GitHub issue per scheduling block, listing all
associated targets and MOUS IDs. Issues are automatically added to the
GitHub Project board and tracked through the data reduction workflow.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from panta_rei.github.project import (
    PROJECT_STATUS_DELIVERED,
    PROJECT_STATUS_IN_PROGRESS,
)

log = logging.getLogger(__name__)

# Default weblog URL base (matches stage_weblogs.py configuration)
DEFAULT_WEBLOG_BASE_URL = "http://www.alma.ac.uk/nas/dwalker2/panta-rei/weblogs"

# Label colors
LABEL_COLORS = {
    "SB": "d73a4a",        # Red - issue type
    "TM": "0e8a16",        # Green - 12m array
    "SM": "1d76db",        # Blue - 7m/ACA
    "TP": "f9a825",        # Amber - Total Power
    "N2H+": "c5def5",      # Light blue - line group
    "HCO+": "d4c5f9",      # Light purple - line group
    "Delivered": "5319e7",  # Purple - status
    "In progress": "fbca04",  # Yellow - status
    "Extracted": "0075ca",  # Blue - status
    "QA-Ready": "2ea44f",   # Green - status
}


@dataclass
class SchedulingBlock:
    """Represents an ALMA Scheduling Block with associated metadata."""
    sb_name: str
    array: str  # TM, SM, TP
    gous_id: str
    mous_ids: List[str] = field(default_factory=list)
    targets: Set[str] = field(default_factory=set)
    line_group: str = ""

    # Status flags (populated from state DB)
    delivered: bool = False
    downloaded: bool = False
    extracted: bool = False
    weblog_path: Optional[Path] = None
    weblog_url: Optional[str] = None

    @property
    def mous_ids_list(self) -> List[str]:
        """Return MOUS IDs as a list."""
        if isinstance(self.mous_ids, str):
            return [m.strip() for m in self.mous_ids.split(";") if m.strip()]
        return list(self.mous_ids)

    def add_target(self, target: str) -> None:
        """Add a target to this SB."""
        if target:
            self.targets.add(target)

    def merge(self, other: SchedulingBlock) -> None:
        """Merge another SB's targets and MOUS IDs into this one."""
        self.targets.update(other.targets)
        for mous in other.mous_ids_list:
            if mous not in self.mous_ids:
                if isinstance(self.mous_ids, list):
                    self.mous_ids.append(mous)
                else:
                    self.mous_ids = self.mous_ids_list + [mous]


def parse_array_from_sb_name(sb_name: str) -> Optional[str]:
    """Extract array type from SB name suffix."""
    sb_lower = sb_name.lower()
    if sb_lower.endswith("_tp"):
        return "TP"
    if sb_lower.endswith("_7m"):
        return "SM"
    if re.search(r"_tm\d*$", sb_lower):
        return "TM"
    return None


def normalize_weblog_url(url: Optional[str]) -> Optional[str]:
    """Normalize a weblog URL to ensure it points to the index.html file.

    Ensures the URL:
    - Uses http:// (not https://)
    - Ends with /html/index.html
    """
    if not url:
        return None

    # Fix protocol: use http instead of https
    if url.startswith("https://"):
        url = "http://" + url[8:]

    # Ensure URL ends with /html/index.html
    url = url.rstrip("/")
    if not url.endswith("/html/index.html"):
        if url.endswith("/html"):
            url += "/index.html"
        elif url.endswith("/index.html"):
            # Already has index.html but missing /html/
            pass
        else:
            url += "/html/index.html"

    return url


def build_sb_issue_body(
    sb: SchedulingBlock,
    weblog_base_url: Optional[str] = None,
    weblog_dir: Optional[Path] = None,
    base_dir: Optional[Path] = None,
) -> str:
    """Build the full GitHub issue body for a scheduling block.

    This is a standalone pure function so it can be used by both the live
    issue manager and the zero-API GH_DRY_RUN path.
    """
    targets_sorted = sorted(sb.targets)
    targets_text = "\n".join(f"  - `{t}`" for t in targets_sorted)

    mous_text = ""
    for mous_id in sb.mous_ids_list:
        full_uid = f"uid://A001/{mous_id.replace('_', '/')}"
        archive_url = (
            f"https://almascience.org/aq/?result_view=observation&mous={full_uid}"
        )
        mous_text += f"  - [`{mous_id}`]({archive_url})\n"

    if sb.weblog_url:
        weblog_line = f"* [x] [Weblog]({sb.weblog_url}) available"
    elif sb.weblog_path:
        if weblog_base_url:
            try:
                if weblog_dir:
                    rel_path = sb.weblog_path.relative_to(weblog_dir)
                elif base_dir:
                    rel_path = sb.weblog_path.relative_to(base_dir)
                else:
                    raise ValueError("no base for relative path")
                weblog_url = f"{weblog_base_url.rstrip('/')}/{rel_path}/index.html"
                weblog_url = normalize_weblog_url(weblog_url)
                weblog_line = f"* [x] [Weblog]({weblog_url}) available"
            except ValueError:
                weblog_line = f"* [x] Weblog available at: `{sb.weblog_path}`"
        else:
            weblog_line = f"* [x] Weblog available at: `{sb.weblog_path}`"
    else:
        weblog_line = "* [ ] Weblog available"

    return f"""## Scheduling Block: {sb.sb_name}

**Array:** {sb.array}
**Line Group:** {sb.line_group or "N/A"}
**GOUS ID:** `{sb.gous_id}`

### Targets

{targets_text}

### MOUS IDs

{mous_text}
---

### Data Status

* [{'x' if sb.delivered else ' '}] Delivered
* [{'x' if sb.downloaded else ' '}] Downloaded
* [{'x' if sb.extracted else ' '}] Extracted
{weblog_line}

### Quality Assessment

* [ ] Weblog reviewed
* [ ] Calibration OK

### Notes

Please add comments to this issue with any details regarding the QA, paying particular attention to e.g., calibration issues, poor continuum identification, size mitigation, clean divergence, etc.
See the QA wiki page for more details: https://github.com/panta-rei-alma/data-reduction/wiki/Weblog-QA-guide
"""


class GitHubIssueManager:
    """Manages GitHub issues for ALMA scheduling blocks."""

    def __init__(
        self,
        project_code: str,
        base_dir: Path,
        gh_owner: str,
        gh_repo: str,
        gh_token: Optional[str] = None,
        weblog_base_url: Optional[str] = None,
        weblog_dir: Optional[Path] = None,
        csv_path: Optional[Path] = None,
        dry_run: bool = False,
        limit: Optional[int] = None,
        gh_project_number: Optional[int] = None,
        update_project_status: bool = False,
        update_targets: bool = False,
    ):
        self.project_code = project_code
        self.base_dir = Path(base_dir)
        self.gh_owner = gh_owner
        self.gh_repo = gh_repo
        self.csv_path = Path(csv_path) if csv_path else None
        self.dry_run = dry_run
        self.limit = limit
        self.weblog_base_url = weblog_base_url or DEFAULT_WEBLOG_BASE_URL
        self.weblog_dir = Path(weblog_dir) if weblog_dir else None
        self.gh_project_number = gh_project_number
        self.update_project_status = update_project_status
        self.update_targets = update_targets

        # Get GitHub token
        self.gh_token = gh_token or os.environ.get("GITHUB_TOKEN")
        if not self.gh_token:
            raise ValueError(
                "GitHub token required. Set GITHUB_TOKEN env var or pass gh_token parameter."
            )

        # Initialize GitHub API (lazy import — ghapi not needed for dry-run)
        from ghapi.all import GhApi
        self.api = GhApi(owner=self.gh_owner, repo=self.gh_repo, token=self.gh_token)

        # Initialize project manager if project number specified
        from panta_rei.github.project import GitHubProjectManager
        self.project_manager: Optional[GitHubProjectManager] = None
        if gh_project_number is not None:
            self.project_manager = GitHubProjectManager(
                token=self.gh_token,
                org=self.gh_owner,
                project_number=gh_project_number,
            )

        # Cache for existing issues
        self._existing_issues: Dict[str, dict] = {}
        self._labels_cache: Set[str] = set()

    def _all_paged(self, api_call, **kwargs) -> List:
        """Flatten paginated API results."""
        from ghapi.all import paged
        return [item for page in paged(api_call, **kwargs) for item in page]

    def load_existing_issues(self) -> Dict[str, dict]:
        """Load all existing issues from the repository."""
        log.info("Loading existing GitHub issues...")
        issues = self._all_paged(self.api.issues.list_for_repo, state="all")

        # Filter out pull requests
        issues = [i for i in issues if not hasattr(i, "pull_request")]
        log.info(f"Found {len(issues)} existing issues")

        # Index by title (SB name should be in title)
        for issue in issues:
            self._existing_issues[issue.title] = issue
            # Also index by just the SB name if it's in the title
            # Title format: "SB: AG231.79_a_03_7M"
            if issue.title.startswith("SB: "):
                sb_name = issue.title[4:].strip()
                self._existing_issues[sb_name] = issue

        return self._existing_issues

    def load_labels(self) -> Set[str]:
        """Load existing labels from the repository."""
        try:
            labels = self._all_paged(self.api.issues.list_labels_for_repo)
            self._labels_cache = {label.name for label in labels}
            log.info(f"Found {len(self._labels_cache)} existing labels")
        except Exception as e:
            log.warning(f"Could not load labels: {e}")
            self._labels_cache = set()
        return self._labels_cache

    def ensure_label(self, label_name: str, color: Optional[str] = None) -> bool:
        """Create a label if it doesn't exist."""
        if label_name in self._labels_cache:
            return True

        if color is None:
            color = LABEL_COLORS.get(label_name, "ededed")

        if self.dry_run:
            log.info(f"[DRY-RUN] Would create label: {label_name}")
            self._labels_cache.add(label_name)
            return True

        try:
            self.api.issues.create_label(name=label_name, color=color)
            self._labels_cache.add(label_name)
            log.info(f"Created label: {label_name}")
            return True
        except Exception as e:
            # Label might already exist (race condition or case mismatch)
            if "already_exists" in str(e).lower():
                self._labels_cache.add(label_name)
                return True
            log.warning(f"Could not create label '{label_name}': {e}")
            return False

    def get_state_from_db(self) -> Dict[str, dict]:
        """Read observation states from the SQLite database."""
        db_path = self.base_dir / "alma_retrieval_state.sqlite3"
        if not db_path.exists():
            log.warning(f"State database not found: {db_path}")
            return {}

        states = {}
        with sqlite3.connect(db_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute("SELECT * FROM obs").fetchall()
            for row in rows:
                states[row["uid"]] = dict(row)

        log.info(f"Loaded {len(states)} observations from state database")
        return states

    def get_weblog_info_from_db(self) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
        """Get weblog paths and URLs from the database.

        Returns:
            Dict mapping canonical UID -> (weblog_path, weblog_url)
        """
        db_path = self.base_dir / "alma_retrieval_state.sqlite3"
        if not db_path.exists():
            return {}

        weblog_info = {}
        try:
            with sqlite3.connect(db_path) as con:
                # Check if weblog columns exist
                cursor = con.execute("PRAGMA table_info(obs)")
                cols = {row[1] for row in cursor.fetchall()}

                if "weblog_path" in cols and "weblog_url" in cols:
                    rows = con.execute(
                        "SELECT uid, weblog_path, weblog_url FROM obs WHERE weblog_staged = 1"
                    ).fetchall()
                    for row in rows:
                        weblog_info[row[0]] = (row[1], row[2])
                    log.info(f"Loaded {len(weblog_info)} staged weblog entries from database")
        except Exception as e:
            log.warning(f"Could not load weblog info from database: {e}")

        return weblog_info

    def find_weblog(
        self,
        sb: SchedulingBlock,
        weblog_info: Optional[Dict[str, Tuple[Optional[str], Optional[str]]]] = None,
    ) -> Tuple[Optional[Path], Optional[str]]:
        """Find weblog directory and URL for a scheduling block.

        Looks up staged weblogs from the database. Weblogs must be staged via
        stage_weblogs.py to appear here.

        Returns:
            Tuple of (weblog_path, weblog_url) - either or both may be None
        """
        if not weblog_info:
            return (None, None)

        for mous_id in sb.mous_ids_list:
            # Convert compact MOUS ID to canonical form for lookup
            mous_lower = mous_id.lower()
            for db_uid, (path, url) in weblog_info.items():
                if mous_lower in db_uid.lower() or mous_id.replace("_", "") in db_uid:
                    weblog_path = Path(path) if path else None
                    # Normalize the URL to ensure correct format
                    weblog_url = normalize_weblog_url(url)
                    return (weblog_path, weblog_url)

        return (None, None)

    def load_sbs_from_csv(self, csv_path: Path) -> Dict[str, SchedulingBlock]:
        """Load scheduling blocks from targets_by_array.csv."""
        if not csv_path.exists():
            log.error(f"CSV file not found: {csv_path}")
            return {}

        sbs: Dict[str, SchedulingBlock] = {}

        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sb_name = row.get("sb_name", "").strip()
                if not sb_name:
                    continue

                if sb_name not in sbs:
                    array = row.get("array", "") or parse_array_from_sb_name(sb_name) or "Unknown"
                    mous_raw = row.get("mous_ids", "")
                    mous_list = [m.strip() for m in mous_raw.split(";") if m.strip()]

                    sbs[sb_name] = SchedulingBlock(
                        sb_name=sb_name,
                        array=array,
                        gous_id=row.get("gous_id", ""),
                        mous_ids=mous_list,
                        line_group=row.get("Line group", ""),
                    )

                # Add target to this SB
                target = row.get("source_name", "").strip()
                sbs[sb_name].add_target(target)

        log.info(f"Loaded {len(sbs)} scheduling blocks from CSV")
        return sbs

    def enrich_sb_status(self, sbs: Dict[str, SchedulingBlock]) -> None:
        """Enrich SBs with status from database and weblog availability."""
        db_states = self.get_state_from_db()
        weblog_info = self.get_weblog_info_from_db()

        for sb_name, sb in sbs.items():
            # Check status for each MOUS ID
            all_extracted = True
            any_downloaded = False
            any_delivered = False

            for mous_id in sb.mous_ids_list:
                # Try to find matching state entry
                # The DB uses full underscore UIDs, we have compact ones
                matching_state = None
                for db_uid, state in db_states.items():
                    if mous_id.lower() in db_uid.lower():
                        matching_state = state
                        break

                if matching_state:
                    status = matching_state.get("status", "")
                    if status == "extracted":
                        any_downloaded = True
                    elif status == "downloaded":
                        any_downloaded = True
                        all_extracted = False
                    else:
                        all_extracted = False

                    # Check release date for delivery status
                    release = matching_state.get("release_date", "")
                    if release and not release.startswith("3000"):
                        any_delivered = True
                else:
                    all_extracted = False

            sb.delivered = any_delivered
            sb.downloaded = any_downloaded
            sb.extracted = all_extracted and any_downloaded

            # Find weblog - check database first, then filesystem
            weblog_path, weblog_url = self.find_weblog(sb, weblog_info)
            sb.weblog_path = weblog_path
            sb.weblog_url = weblog_url

    def get_project_board_status(self, sb: SchedulingBlock) -> str:
        """Determine the appropriate project board status for a scheduling block.

        Returns the status column name based on the SB's delivery state.
        """
        if sb.delivered:
            return PROJECT_STATUS_DELIVERED
        else:
            return PROJECT_STATUS_IN_PROGRESS

    def build_issue_body(self, sb: SchedulingBlock) -> str:
        """Build the issue body with status checkboxes."""
        return build_sb_issue_body(
            sb,
            weblog_base_url=self.weblog_base_url,
            weblog_dir=self.weblog_dir,
            base_dir=self.base_dir,
        )

    def create_issue(self, sb: SchedulingBlock) -> Optional[dict]:
        """Create a GitHub issue for a scheduling block."""
        title = f"SB: {sb.sb_name}"

        # Check if issue already exists
        if title in self._existing_issues or sb.sb_name in self._existing_issues:
            log.debug(f"Issue already exists for {sb.sb_name}")
            return None

        body = self.build_issue_body(sb)

        # Determine labels
        labels = ["SB"]
        self.ensure_label("SB")

        # Add array type label
        if sb.array:
            self.ensure_label(sb.array)
            labels.append(sb.array)

        # Add line group label
        if sb.line_group:
            self.ensure_label(sb.line_group)
            labels.append(sb.line_group)

        # Add target labels
        for target in sorted(sb.targets):
            self.ensure_label(target)
            labels.append(target)

        # Add status labels based on actual delivery state
        if sb.delivered:
            self.ensure_label("Delivered")
            labels.append("Delivered")
        else:
            self.ensure_label("In progress")
            labels.append("In progress")

        if sb.extracted:
            self.ensure_label("Extracted")
            labels.append("Extracted")

        # Determine project board status
        project_status = self.get_project_board_status(sb)

        if self.dry_run:
            log.info(f"[DRY-RUN] Would create issue: {title}")
            log.info(f"  Targets: {len(sb.targets)}, MOUS: {sb.mous_ids_list}")
            log.info(f"  Labels: {labels}")
            log.info(f"  Delivered: {sb.delivered}")
            if sb.weblog_url:
                log.info(f"  Weblog URL: {sb.weblog_url}")
            if self.project_manager:
                log.info(f"  Project status: {project_status}")
            return {"title": title, "labels": labels, "dry_run": True}

        try:
            issue = self.api.issues.create(
                title=title,
                body=body,
                labels=labels,
            )
            log.info(f"Created issue #{issue.number}: {title}")
            self._existing_issues[title] = issue
            self._existing_issues[sb.sb_name] = issue

            # Add to project board if configured
            if self.project_manager:
                self._add_issue_to_project(issue, sb)

            return issue
        except Exception as e:
            log.error(f"Failed to create issue for {sb.sb_name}: {e}")
            return None

    def _add_issue_to_project(self, issue: Any, sb: SchedulingBlock) -> bool:
        """Add an issue to the project board with appropriate status.

        Status is determined by delivery state:
        - "In progress" for SBs not yet delivered
        - "Delivered" for SBs that have been delivered
        """
        if not self.project_manager:
            return False

        # Get the issue's node ID
        node_id = self.project_manager.get_issue_node_id(
            self.gh_owner, self.gh_repo, issue.number,
        )
        if not node_id:
            log.warning(f"Could not get node ID for issue #{issue.number}")
            return False

        # Add to project
        item_id = self.project_manager.add_issue_to_project(node_id)
        if not item_id:
            log.warning(f"Could not add issue #{issue.number} to project")
            return False

        # Determine status based on delivery state
        status = self.get_project_board_status(sb)

        # Set the status
        if self.project_manager.set_item_status(item_id, status):
            log.info(f"  Added to project board with status: {status}")
            return True
        else:
            log.warning(f"  Added to project but could not set status")
            return False

    def update_issue(self, sb: SchedulingBlock) -> bool:
        """Update an existing issue with current status."""
        existing = self._existing_issues.get(f"SB: {sb.sb_name}") or \
                   self._existing_issues.get(sb.sb_name)

        if not existing:
            return False

        body = existing.body
        current_labels = {label.name for label in existing.labels} if hasattr(existing, 'labels') else set()
        new_labels = set(current_labels)

        needs_update = False

        # Update targets section if requested
        if self.update_targets:
            targets_sorted = sorted(sb.targets)
            new_targets_text = "\n".join(f"  - `{t}`" for t in targets_sorted)

            # Find and replace the targets section (between "### Targets" and "### MOUS IDs")
            targets_pattern = r"(### Targets\n\n)(.*?)((\n\n)?### MOUS IDs)"
            match = re.search(targets_pattern, body, re.DOTALL)
            if match:
                old_targets = match.group(2).strip()
                # Check if the section is different (ignoring whitespace differences)
                if old_targets != new_targets_text.strip():
                    body = re.sub(
                        targets_pattern,
                        rf"\g<1>{new_targets_text}\n\n### MOUS IDs",
                        body,
                        flags=re.DOTALL
                    )
                    needs_update = True
                    log.debug(f"Updated targets section for {sb.sb_name}")

        # Ensure target labels are present
        for target in sorted(sb.targets):
            self.ensure_label(target)
            new_labels.add(target)

        if new_labels != current_labels:
            needs_update = True

        # Check and update delivery status (both checkbox and label)
        if sb.delivered:
            if "[ ] Delivered" in body:
                body = body.replace("[ ] Delivered", "[x] Delivered")
                needs_update = True
            if "In progress" in new_labels:
                new_labels.remove("In progress")
                needs_update = True
            if "Delivered" not in new_labels:
                new_labels.add("Delivered")
                self.ensure_label("Delivered")
                needs_update = True
        else:
            # Not delivered - ensure "In progress" label
            if "Delivered" in new_labels:
                new_labels.remove("Delivered")
                needs_update = True
            if "In progress" not in new_labels:
                new_labels.add("In progress")
                self.ensure_label("In progress")
                needs_update = True

        if sb.downloaded and "[ ] Downloaded" in body:
            body = body.replace("[ ] Downloaded", "[x] Downloaded")
            needs_update = True

        if sb.extracted and "[ ] Extracted" in body:
            body = body.replace("[ ] Extracted", "[x] Extracted")
            new_labels.add("Extracted")
            self.ensure_label("Extracted")
            needs_update = True

        # Update weblog link - prefer weblog_url from staging (already normalized)
        if (sb.weblog_url or sb.weblog_path) and "[ ] Weblog available" in body:
            if sb.weblog_url:
                body = body.replace("[ ] Weblog available", f"[x] [Weblog]({sb.weblog_url}) available")
            elif sb.weblog_path:
                if self.weblog_base_url:
                    try:
                        if self.weblog_dir:
                            rel_path = sb.weblog_path.relative_to(self.weblog_dir)
                        else:
                            rel_path = sb.weblog_path.relative_to(self.base_dir)
                        weblog_url = f"{self.weblog_base_url.rstrip('/')}/{rel_path}/index.html"
                        weblog_url = normalize_weblog_url(weblog_url)
                        body = body.replace("[ ] Weblog available", f"[x] [Weblog]({weblog_url}) available")
                    except ValueError:
                        body = body.replace("[ ] Weblog available", f"[x] Weblog available at: `{sb.weblog_path}`")
                else:
                    body = body.replace("[ ] Weblog available", f"[x] Weblog available at: `{sb.weblog_path}`")
            needs_update = True

        # Update project board status if enabled
        project_status_updated = False
        if self.update_project_status and self.project_manager:
            project_status_updated = self._update_project_status(existing, sb)

        if not needs_update and not project_status_updated:
            return False

        if self.dry_run:
            log.info(f"[DRY-RUN] Would update issue #{existing.number}: {existing.title}")
            if project_status_updated:
                log.info(f"  Would update project status to: {self.get_project_board_status(sb)}")
            return True

        if needs_update:
            try:
                self.api.issues.update(
                    issue_number=existing.number,
                    body=body,
                    labels=list(new_labels),
                )
                log.info(f"Updated issue #{existing.number}: {existing.title}")
            except Exception as e:
                log.error(f"Failed to update issue #{existing.number}: {e}")
                return False

        return True

    def _update_project_status(self, issue: Any, sb: SchedulingBlock) -> bool:
        """Update the project board status for an existing issue.

        Only updates if the issue is in "In progress" or "Delivered" status.
        Does not touch issues that have been manually advanced to review states.
        """
        if not self.project_manager:
            return False

        # Get the issue's node ID
        node_id = self.project_manager.get_issue_node_id(
            self.gh_owner, self.gh_repo, issue.number,
        )
        if not node_id:
            return False

        # Check current status
        current_status = self.project_manager.get_current_status(node_id)

        # Don't update if manually advanced to a review/work state
        if self.project_manager.is_status_manually_advanced(current_status):
            log.debug(f"Issue #{issue.number} is in '{current_status}', not updating")
            return False

        # Get the appropriate status based on delivery state
        new_status = self.get_project_board_status(sb)

        # Only update if status has changed
        if current_status == new_status:
            return False

        # Get or add the item to the project
        item_id = self.project_manager._existing_items.get(node_id)
        if not item_id:
            item_id = self.project_manager.add_issue_to_project(node_id)

        if not item_id:
            return False

        if self.dry_run:
            log.info(f"[DRY-RUN] Would update project status for #{issue.number}: {current_status} -> {new_status}")
            return True

        if self.project_manager.set_item_status(item_id, new_status):
            log.info(f"Updated project status for #{issue.number}: {current_status} -> {new_status}")
            return True

        return False

    def run(self) -> Tuple[int, int]:
        """Main entry point: load SBs from CSV, create/update issues.

        Returns:
            Tuple of (created_count, updated_count)
        """
        # Load existing state
        self.load_existing_issues()
        self.load_labels()

        # Load project metadata if using project board
        if self.project_manager:
            if not self.project_manager.load_project_metadata():
                log.warning("Could not load project metadata, issues won't be added to project board")
                self.project_manager = None

        # Load SBs from CSV (use explicit csv_path if set, else default)
        csv_path = self.csv_path or (self.base_dir / "targets_by_array.csv")
        sbs = self.load_sbs_from_csv(csv_path)

        if not sbs:
            log.warning("No scheduling blocks found")
            return 0, 0

        # Enrich with status information
        self.enrich_sb_status(sbs)

        # Apply limit if specified
        sb_items = sorted(sbs.items())
        if self.limit is not None and self.limit > 0:
            sb_items = sb_items[:self.limit]
            log.info(f"Processing {len(sb_items)} scheduling blocks (limited from {len(sbs)})")
        else:
            log.info(f"Processing {len(sb_items)} scheduling blocks")

        created = 0
        updated = 0

        for sb_name, sb in sb_items:
            # Try to create new issue
            result = self.create_issue(sb)
            if result:
                created += 1
            else:
                # Try to update existing issue
                if self.update_issue(sb):
                    updated += 1

        log.info(f"Summary: {created} created, {updated} updated")
        return created, updated
