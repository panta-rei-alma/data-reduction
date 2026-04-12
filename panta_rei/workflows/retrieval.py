"""Daily automated retrieval workflow.

Four steps executed in sequence:

1. **RetrieveStep** -- download and extract new MOUSes from the ALMA archive.
2. **BuildTableStep** -- query ALMA metadata and write ``targets_by_array.csv``.
3. **StageWeblogsStep** -- extract weblogs to public directory.
4. **UpdateIssuesStep** -- create / update GitHub issues per scheduling block.

Error semantics (from the migration plan):

* Retrieve failure is **fatal only** when no local data exists (i.e. no
  ``science_goal.*`` directories under ``data_dir``).
* Build-table failure blocks the issues step but **not** weblogs.
* Stage-weblogs failure is non-blocking.
* Issues step is terminal -- its failure is recorded but does not affect
  earlier results.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from panta_rei.workflows.base import Step, StepResult, WorkflowContext

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Retrieve
# ---------------------------------------------------------------------------

class RetrieveStep(Step):
    """Download new MOUSes from the ALMA archive and extract tarballs."""

    @property
    def name(self) -> str:
        return "retrieve"

    @property
    def description(self) -> str:
        return "Download and extract ALMA data"

    def should_skip(self, ctx: WorkflowContext) -> str | None:
        base = super().should_skip(ctx)
        if base is not None:
            return base
        if ctx.username is None:
            return "no ALMA username provided"
        return None

    def run(self, ctx: WorkflowContext) -> StepResult:
        from panta_rei.alma.download import retrieve_and_extract

        try:
            retrieve_and_extract(
                username=ctx.username,
                project_code=ctx.config.project_code,
                base_dir=ctx.data_dir,
                db_manager=ctx.db_manager,
            )
            return StepResult(
                success=True,
                summary="Retrieval completed",
                items_processed=1,
            )
        except Exception as exc:
            return StepResult(
                success=False,
                summary=f"Retrieval failed: {exc}",
                errors=[str(exc)],
            )


# ---------------------------------------------------------------------------
# Step 2: Build table
# ---------------------------------------------------------------------------

class BuildTableStep(Step):
    """Query ALMA metadata and write targets_by_array.csv."""

    @property
    def name(self) -> str:
        return "build_table"

    @property
    def description(self) -> str:
        return "Query ALMA metadata and build targets CSV"

    def run(self, ctx: WorkflowContext) -> StepResult:
        from panta_rei.alma.metadata import (
            build_gous_to_sgous_map,
            build_index,
            query_rows,
            write_csv,
        )

        try:
            rows = query_rows(ctx.config.project_code, ctx.username)
            if not rows:
                return StepResult(
                    success=True,
                    summary="No metadata rows returned from ALMA",
                    items_processed=0,
                )

            agg = build_index(rows)
            gous_map = build_gous_to_sgous_map(ctx.data_dir)
            write_csv(agg, ctx.targets_csv_path, gous_map)

            return StepResult(
                success=True,
                summary=(
                    f"Wrote {len(agg)} entries to {ctx.targets_csv_path.name}"
                ),
                items_processed=len(agg),
            )
        except Exception as exc:
            return StepResult(
                success=False,
                summary=f"Build table failed: {exc}",
                errors=[str(exc)],
            )


# ---------------------------------------------------------------------------
# Step 3: Stage weblogs
# ---------------------------------------------------------------------------

class StageWeblogsStep(Step):
    """Extract weblogs to the public-facing weblog directory."""

    @property
    def name(self) -> str:
        return "stage_weblogs"

    @property
    def description(self) -> str:
        return "Stage weblogs for public web access"

    def run(self, ctx: WorkflowContext) -> StepResult:
        from panta_rei.alma.staging import WeblogStager, WeblogStateDB

        try:
            db_path = ctx.config.state_db_path
            weblog_db = WeblogStateDB(db_path) if db_path.exists() else None

            stager = WeblogStager(
                base_dir=ctx.base_dir,
                weblog_dir=ctx.config.weblog_dir,
                db=weblog_db,
                url_mappings=ctx.config.url_mappings,
                dry_run=ctx.dry_run,
            )
            staged = stager.stage_all()
            staged_count = len(staged)
            corrupted = stager.corrupted_count

            parts = [f"Staged {staged_count} weblog(s)"]
            if corrupted:
                parts.append(f"{corrupted} corrupted reset for re-download")

            return StepResult(
                success=True,
                summary="; ".join(parts),
                items_processed=staged_count,
                items_skipped=corrupted,
            )
        except Exception as exc:
            return StepResult(
                success=False,
                summary=f"Weblog staging failed: {exc}",
                errors=[str(exc)],
            )


# ---------------------------------------------------------------------------
# Step 4: Update GitHub issues
# ---------------------------------------------------------------------------

class UpdateIssuesStep(Step):
    """Create or update GitHub issues per scheduling block."""

    @property
    def name(self) -> str:
        return "update_issues"

    @property
    def description(self) -> str:
        return "Create/update GitHub issues for scheduling blocks"

    def should_skip(self, ctx: WorkflowContext) -> str | None:
        base = super().should_skip(ctx)
        if base is not None:
            return base
        if not ctx.targets_csv_path.exists():
            return f"targets CSV not found: {ctx.targets_csv_path}"
        return None

    def run(self, ctx: WorkflowContext) -> StepResult:
        gh_dry_run = bool(os.environ.get("GH_DRY_RUN"))
        dry_run = ctx.dry_run or gh_dry_run

        # GH_DRY_RUN=1 means zero API calls — generate payloads from local
        # data only.  We do NOT require a token or construct live clients.
        if gh_dry_run:
            return self._run_zero_api_dry_run(ctx)

        from panta_rei.auth import resolve_github_token

        gh_token = resolve_github_token()
        if gh_token is None:
            return StepResult(
                success=False,
                summary="GitHub token not available",
                errors=[
                    "Set GITHUB_TOKEN env var or provide a systemd credential"
                ],
            )

        try:
            from panta_rei.github.issues import GitHubIssueManager

            manager = GitHubIssueManager(
                project_code=ctx.config.project_code,
                base_dir=ctx.base_dir,
                gh_owner=ctx.config.gh_owner,
                gh_repo=ctx.config.gh_repo,
                gh_token=gh_token,
                weblog_dir=ctx.config.weblog_dir,
                csv_path=ctx.targets_csv_path,
                dry_run=dry_run,
                gh_project_number=ctx.config.gh_project_number,
                update_project_status=True,
                update_targets=True,
            )
            created, updated = manager.run()

            return StepResult(
                success=True,
                summary=f"Issues: {created} created, {updated} updated",
                items_processed=created + updated,
            )
        except Exception as exc:
            return StepResult(
                success=False,
                summary=f"Issue update failed: {exc}",
                errors=[str(exc)],
            )

    def _run_zero_api_dry_run(self, ctx: WorkflowContext) -> StepResult:
        """Generate full issue payloads from local CSV+DB with zero API calls.

        Produces the same titles, bodies, labels, and project-status values
        that the live path would, but reads only from the local CSV and
        state DB.  Output is written to a JSON log file for comparison
        against a baseline from the old code (plan Stage A).
        """
        import csv as csv_mod
        import json
        import sqlite3

        csv_path = ctx.targets_csv_path
        if not csv_path.exists():
            return StepResult(
                success=False,
                summary=f"GH_DRY_RUN: CSV not found at {csv_path}",
                errors=["targets CSV missing"],
            )

        # --- Load SBs from CSV (same logic as GitHubIssueManager) ----------
        from panta_rei.github.issues import (
            SchedulingBlock,
            build_sb_issue_body,
            normalize_weblog_url,
            parse_array_from_sb_name,
        )
        from panta_rei.github.project import (
            PROJECT_STATUS_DELIVERED,
            PROJECT_STATUS_IN_PROGRESS,
        )

        sbs: dict[str, SchedulingBlock] = {}
        with open(csv_path, newline="") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                sb_name = row.get("sb_name", "").strip()
                if not sb_name:
                    continue
                if sb_name not in sbs:
                    array = (
                        row.get("array", "")
                        or parse_array_from_sb_name(sb_name)
                        or "Unknown"
                    )
                    mous_raw = row.get("mous_ids", "")
                    mous_list = [
                        m.strip() for m in mous_raw.split(";") if m.strip()
                    ]
                    sbs[sb_name] = SchedulingBlock(
                        sb_name=sb_name,
                        array=array,
                        gous_id=row.get("gous_id", ""),
                        mous_ids=mous_list,
                        line_group=row.get("Line group", ""),
                    )
                target = row.get("source_name", "").strip()
                sbs[sb_name].add_target(target)

        # --- Enrich from DB (status, weblog info) --------------------------
        db_states: dict[str, dict] = {}
        weblog_info: dict[str, tuple] = {}
        db_path = ctx.config.state_db_path
        if db_path.exists():
            con = sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True
            )
            con.row_factory = sqlite3.Row
            for row in con.execute("SELECT * FROM obs"):
                db_states[row["uid"]] = dict(row)
            # Weblog info
            cols = {r[1] for r in con.execute("PRAGMA table_info(obs)")}
            if "weblog_path" in cols and "weblog_url" in cols:
                for row in con.execute(
                    "SELECT uid, weblog_path, weblog_url "
                    "FROM obs WHERE weblog_staged = 1"
                ):
                    weblog_info[row[0]] = (row[1], row[2])
            con.close()

        # --- Build payloads ------------------------------------------------
        payloads: list[dict] = []
        for sb_name, sb in sorted(sbs.items()):
            # Derive delivery/extraction status from DB
            any_delivered = False
            any_downloaded = False
            all_extracted = True
            for mous_id in sb.mous_ids_list:
                match = None
                for db_uid, state in db_states.items():
                    if mous_id.lower() in db_uid.lower():
                        match = state
                        break
                if match:
                    status = match.get("status", "")
                    if status == "extracted":
                        any_downloaded = True
                    elif status == "downloaded":
                        any_downloaded = True
                        all_extracted = False
                    else:
                        all_extracted = False
                    rel = match.get("release_date", "")
                    if rel and not str(rel).startswith("3000"):
                        any_delivered = True
                else:
                    all_extracted = False

            sb.delivered = any_delivered
            sb.downloaded = any_downloaded
            sb.extracted = all_extracted and any_downloaded

            # Weblog URL
            for mous_id in sb.mous_ids_list:
                for db_uid, (path, url) in weblog_info.items():
                    if mous_id.lower() in db_uid.lower():
                        sb.weblog_url = normalize_weblog_url(url)
                        break
                if sb.weblog_url:
                    break

            # Labels
            labels = ["SB"]
            if sb.array:
                labels.append(sb.array)
            if sb.line_group:
                labels.append(sb.line_group)
            for t in sorted(sb.targets):
                labels.append(t)
            labels.append("Delivered" if sb.delivered else "In progress")
            if sb.extracted:
                labels.append("Extracted")

            # Project board status
            project_status = (
                PROJECT_STATUS_DELIVERED
                if sb.delivered
                else PROJECT_STATUS_IN_PROGRESS
            )

            # Full-fidelity issue body via the same builder the live path uses
            body = build_sb_issue_body(sb)

            payloads.append({
                "title": f"SB: {sb_name}",
                "body": body,
                "labels": labels,
                "project_status": project_status,
                "array": sb.array,
                "source_names": sorted(sb.targets),
                "gous_id": sb.gous_id,
                "mous_ids": sb.mous_ids_list,
                "line_group": sb.line_group,
                "delivered": sb.delivered,
                "extracted": sb.extracted,
                "weblog_url": sb.weblog_url,
            })

        # Write payloads
        log_path = csv_path.parent / "gh_dry_run_payloads.json"
        with open(log_path, "w") as f:
            json.dump(payloads, f, indent=2)

        log.info(
            "GH_DRY_RUN: Generated %d full issue payloads -> %s "
            "(zero API calls)",
            len(payloads),
            log_path,
        )

        return StepResult(
            success=True,
            summary=f"GH_DRY_RUN: {len(payloads)} payloads written to {log_path.name}",
            items_processed=len(payloads),
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _has_local_data(data_dir: Path) -> bool:
    """Return True if *data_dir* contains at least one science_goal.* dir."""
    if not data_dir.is_dir():
        return False
    return any(
        p.is_dir() and p.name.startswith("science_goal.")
        for p in data_dir.iterdir()
    )


def run_retrieval(ctx: WorkflowContext) -> dict[str, StepResult]:
    """Run the four-step retrieval pipeline with dependency-aware error handling.

    Error semantics:
    * Retrieve failure is fatal only when no local data exists.
    * Build-table failure blocks issue updates but not weblog staging.
    * Weblog staging failure is non-blocking.
    * Issue step is terminal.

    Returns
    -------
    dict mapping step name to :class:`StepResult`.
    """
    results: dict[str, StepResult] = {}

    # -- Step 1: Retrieve ---------------------------------------------------
    retrieve = RetrieveStep()
    skip = retrieve.should_skip(ctx)
    if skip is not None:
        log.info("Skipping %s: %s", retrieve.name, skip)
        results[retrieve.name] = StepResult(success=True, summary=f"Skipped: {skip}")
    else:
        log.info("--- Running step: %s (%s) ---", retrieve.name, retrieve.description)
        result = retrieve.run(ctx)
        results[retrieve.name] = result

        if not result.success:
            if _has_local_data(ctx.data_dir):
                log.warning(
                    "Retrieve failed but local data exists -- continuing "
                    "with existing data: %s",
                    result.summary,
                )
            else:
                log.error(
                    "Retrieve failed and no local data found -- aborting: %s",
                    result.summary,
                )
                return results

    # -- Step 2: Build table ------------------------------------------------
    build = BuildTableStep()
    skip = build.should_skip(ctx)
    build_ok = False
    if skip is not None:
        log.info("Skipping %s: %s", build.name, skip)
        results[build.name] = StepResult(success=True, summary=f"Skipped: {skip}")
        # If an existing CSV is present, the issues step can still run.
        build_ok = ctx.targets_csv_path.exists()
    else:
        log.info("--- Running step: %s (%s) ---", build.name, build.description)
        result = build.run(ctx)
        results[build.name] = result
        build_ok = result.success

        if not result.success:
            log.warning(
                "Build table failed -- weblog staging will proceed, "
                "but issue updates will be skipped: %s",
                result.summary,
            )

    # -- Step 3: Stage weblogs (non-blocking) -------------------------------
    stage = StageWeblogsStep()
    skip = stage.should_skip(ctx)
    if skip is not None:
        log.info("Skipping %s: %s", stage.name, skip)
        results[stage.name] = StepResult(success=True, summary=f"Skipped: {skip}")
    else:
        log.info("--- Running step: %s (%s) ---", stage.name, stage.description)
        result = stage.run(ctx)
        results[stage.name] = result

        if not result.success:
            log.warning(
                "Weblog staging failed (non-blocking): %s", result.summary
            )

    # -- Step 4: Update issues (requires build_ok) --------------------------
    issues = UpdateIssuesStep()
    skip = issues.should_skip(ctx)
    if skip is not None:
        log.info("Skipping %s: %s", issues.name, skip)
        results[issues.name] = StepResult(success=True, summary=f"Skipped: {skip}")
    elif not build_ok:
        msg = "Skipped because build_table did not succeed and no CSV available"
        log.info("Skipping %s: %s", issues.name, msg)
        results[issues.name] = StepResult(success=True, summary=msg)
    else:
        log.info("--- Running step: %s (%s) ---", issues.name, issues.description)
        result = issues.run(ctx)
        results[issues.name] = result

    return results
