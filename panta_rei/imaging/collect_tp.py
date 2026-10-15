"""Collect standalone total-power (TP) cubes into a flat per-member tree.

The ALMA archive delivers TP single-dish cubes inside the per-MOUS
delivery tree::

    <data_dir>/science_goal.uid___A001_<sgous>/group.uid___A001_<gous>/
        member.uid___A001_<mous>/product/
            member.uid___A001_<mous>.<source>_sci.spw<N>.cube.I.sd.fits

Those cubes are consumed by the feather step (see
:func:`panta_rei.imaging.matching.find_tp_cube`) but are never republished
on their own, so there is no single place to inspect them. This module
builds one: a ``total_power/member.uid___A001_<mous>/`` tree of **hard
links** to the delivered cubes.

Hard links are used deliberately — the collected tree costs no additional
disk, and because a link is just a second name for the same inode there is
no risk of the QA copy drifting from the delivered original. Filenames are
left exactly as delivered: these cubes are for QA and research only, not
for re-delivery to the ALMA Science Archive, so the strict archive naming
convention applied to the feathered 12m+7m+TP products does not apply here
and plain names keep provenance obvious.

Only ``*.cube.I.sd.fits`` is collected. The companion ``*.sd.weight.fits``
cubes and the leftover CASA-native ``*.sd.image`` directories are ignored
(the latter could not be hard-linked in any case — links do not work on
directories).
"""

from __future__ import annotations

import csv
import logging
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from panta_rei.imaging.matching import (
    find_member_dir,
    load_targets_csv,
    sanitize_source_name_for_glob,
)

logger = logging.getLogger(__name__)

#: Only single-dish science cubes are collected — never the weight cubes.
TP_CUBE_GLOB_SUFFIX = "*spw*.cube.I.sd.fits"

#: Directory created under the project dir to hold the collected tree.
DEFAULT_COLLECT_DIRNAME = "total_power"

#: Manifest filename written at the root of the collected tree.
MANIFEST_NAME = "tp_manifest.csv"

_SPW_RE = re.compile(r"spw(\d+)")
_FREQ_RANGE_RE = re.compile(r"\.(\d+\.\d+)-(\d+\.\d+)GHz\.")

MANIFEST_COLUMNS = [
    "member_mous",
    "sgous_id",
    "gous_id",
    "source_name",
    "line_group",
    "spw",
    "freq_min_ghz",
    "freq_max_ghz",
    "filename",
    "link_path",
    "source_path",
    "size_bytes",
    "inode",
]


@dataclass(frozen=True)
class TPCube:
    """One delivered TP cube and the metadata needed to place/describe it."""

    path: Path
    mous_id: str
    sgous_id: str
    gous_id: str
    source_name: str
    line_group: str
    spw: Optional[int]

    @property
    def member_dirname(self) -> str:
        return f"member.uid___A001_{self.mous_id}"


@dataclass
class CollectStats:
    """Outcome tallies for a collection run."""

    linked: int = 0
    already: int = 0
    replaced: int = 0
    conflict: int = 0
    failed: int = 0

    @property
    def total_present(self) -> int:
        """Links that exist and point at the delivered inode."""
        return self.linked + self.already + self.replaced


def _parse_spw(name: str) -> Optional[int]:
    m = _SPW_RE.search(name)
    return int(m.group(1)) if m else None


def discover_tp_cubes(
    data_dir: Path,
    csv_path: Path,
    *,
    member: Optional[Iterable[str]] = None,
    source: Optional[Iterable[str]] = None,
) -> list[TPCube]:
    """Find every delivered TP cube described by ``targets_by_array.csv``.

    Drives discovery from the CSV (the authoritative list of which MOUSes
    are TP) rather than a blind filesystem walk, so a cube that exists on
    disk but is not described by the CSV is deliberately not collected.

    Mirrors :func:`panta_rei.imaging.matching.find_tp_cube`: the source
    token in delivered filenames has ``+`` replaced by ``p`` but keeps a
    literal ``-``, so the glob must use
    :func:`~panta_rei.imaging.matching.sanitize_source_name_for_glob` and
    never the output-name sanitiser.
    """
    member_filters = [m.lower() for m in member] if member else None
    source_filters = [s.lower() for s in source] if source else None

    groups = load_targets_csv(csv_path)
    seen_pairs: set[tuple[str, str]] = set()
    missing_members: list[str] = []
    empty_members: list[str] = []
    cubes: list[TPCube] = []

    for gous_id, targets in sorted(groups.items()):
        for tg in targets:
            if tg.array != "TP":
                continue
            for mous_id in tg.mous_ids:
                pair = (mous_id, tg.source_name)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                if member_filters and not any(
                    f in mous_id.lower() for f in member_filters
                ):
                    continue
                if source_filters and not any(
                    f in tg.source_name.lower() for f in source_filters
                ):
                    continue

                member_dir = find_member_dir(data_dir, mous_id)
                if member_dir is None:
                    missing_members.append(mous_id)
                    continue
                product_dir = member_dir / "product"
                if not product_dir.is_dir():
                    missing_members.append(mous_id)
                    continue

                source_glob = sanitize_source_name_for_glob(tg.source_name)
                pattern = f"*{source_glob}{TP_CUBE_GLOB_SUFFIX}"
                found = sorted(product_dir.glob(pattern))
                if not found:
                    empty_members.append(f"{mous_id}/{tg.source_name}")
                    continue

                for fits_path in found:
                    cubes.append(TPCube(
                        path=fits_path,
                        mous_id=mous_id,
                        sgous_id=tg.sgous_id,
                        gous_id=gous_id,
                        source_name=tg.source_name,
                        line_group=tg.line_group,
                        spw=_parse_spw(fits_path.name),
                    ))

    if missing_members:
        logger.warning(
            "%d TP MOUS had no product dir on disk: %s",
            len(missing_members), ", ".join(sorted(set(missing_members))),
        )
    if empty_members:
        logger.warning(
            "%d (MOUS, source) pairs matched no TP cube: %s",
            len(empty_members), ", ".join(sorted(set(empty_members))),
        )

    cubes.sort(key=lambda c: (c.mous_id, c.source_name, c.spw or -1))
    logger.info(
        "Discovered %d TP cubes across %d members (%d sources)",
        len(cubes),
        len({c.mous_id for c in cubes}),
        len({c.source_name for c in cubes}),
    )
    return cubes


def load_freq_ranges(imaging_db: Path) -> dict[str, tuple[str, str]]:
    """Map delivered TP cube path -> ``(freq_min_ghz, freq_max_ghz)``.

    Best-effort enrichment for the manifest only. The frequency token is
    lifted from the feathered product each TP cube fed, so the manifest
    agrees with the published ``imaging/output`` names rather than
    inventing a third convention (the token is derived from TM recovered
    params there, from the SM header in the legacy ``combined`` tree, and
    would differ again if recomputed from the TP header here).

    Never raises: a missing or unreadable DB simply yields no enrichment.
    """
    if not imaging_db.is_file():
        logger.warning("imaging DB not found, manifest will omit frequencies: %s", imaging_db)
        return {}

    freqs: dict[str, tuple[str, str]] = {}
    try:
        uri = f"file:{imaging_db}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            rows = conn.execute(
                "SELECT sdimage, output_fits FROM imaging_runs "
                "WHERE status = 'success' AND sdimage IS NOT NULL AND sdimage != '' "
                "ORDER BY rowid"
            ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("could not read imaging DB (%s), manifest will omit frequencies", exc)
        return {}

    for sdimage, output_fits in rows:
        if not output_fits:
            continue
        m = _FREQ_RANGE_RE.search(os.path.basename(output_fits))
        if m:
            # Later rows win: a re-run's token supersedes an earlier one.
            freqs[sdimage] = (m.group(1), m.group(2))

    logger.info("Loaded frequency ranges for %d TP cubes from %s", len(freqs), imaging_db.name)
    return freqs


def link_cube(
    cube: TPCube,
    dest_dir: Path,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> tuple[str, Path]:
    """Hard-link one cube into ``dest_dir``. Returns ``(status, link_path)``.

    Status is one of ``linked``, ``already``, ``replaced``, ``conflict``
    or ``failed``. Idempotent: an existing link to the same inode is left
    untouched, so re-running is a cheap no-op.
    """
    member_dir = dest_dir / cube.member_dirname
    link_path = member_dir / cube.path.name

    if link_path.exists():
        try:
            same = link_path.stat().st_ino == cube.path.stat().st_ino
        except OSError as exc:
            logger.error("%s: could not stat existing link: %s", link_path.name, exc)
            return "failed", link_path
        if same:
            return "already", link_path
        if not force:
            logger.warning(
                "%s exists but points at a different inode (use --force to replace)",
                link_path,
            )
            return "conflict", link_path
        if not dry_run:
            try:
                link_path.unlink()
            except OSError as exc:
                logger.error("%s: could not remove stale link: %s", link_path.name, exc)
                return "failed", link_path
        status = "replaced"
    else:
        status = "linked"

    if dry_run:
        return status, link_path

    try:
        member_dir.mkdir(parents=True, exist_ok=True)
        os.link(cube.path, link_path)
    except OSError as exc:
        logger.error("%s: hard link failed: %s", cube.path.name, exc)
        return "failed", link_path

    return status, link_path


def write_manifest(
    rows: list[dict],
    dest_dir: Path,
    *,
    dry_run: bool = False,
) -> Path:
    """Write the provenance manifest at the root of the collected tree."""
    manifest_path = dest_dir / MANIFEST_NAME
    if dry_run:
        logger.info("[dry-run] would write manifest with %d rows: %s", len(rows), manifest_path)
        return manifest_path

    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_suffix(f".csv.tmp.{os.getpid()}")
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, manifest_path)
    logger.info("Wrote manifest (%d rows): %s", len(rows), manifest_path)
    return manifest_path


def collect(
    cubes: list[TPCube],
    dest_dir: Path,
    *,
    freq_ranges: Optional[dict[str, tuple[str, str]]] = None,
    force: bool = False,
    dry_run: bool = False,
) -> tuple[CollectStats, list[dict]]:
    """Link every cube into ``dest_dir`` and build the manifest rows."""
    freq_ranges = freq_ranges or {}
    stats = CollectStats()
    rows: list[dict] = []

    for cube in cubes:
        status, link_path = link_cube(cube, dest_dir, force=force, dry_run=dry_run)
        setattr(stats, status, getattr(stats, status) + 1)
        if status in ("conflict", "failed"):
            continue

        freq_min, freq_max = freq_ranges.get(str(cube.path), ("", ""))
        try:
            st = cube.path.stat()
            size, inode = st.st_size, st.st_ino
        except OSError:
            size, inode = "", ""

        rows.append({
            "member_mous": cube.mous_id,
            "sgous_id": cube.sgous_id,
            "gous_id": cube.gous_id,
            "source_name": cube.source_name,
            "line_group": cube.line_group,
            "spw": cube.spw if cube.spw is not None else "",
            "freq_min_ghz": freq_min,
            "freq_max_ghz": freq_max,
            "filename": cube.path.name,
            "link_path": str(link_path.relative_to(dest_dir)),
            "source_path": str(cube.path),
            "size_bytes": size,
            "inode": inode,
        })

    return stats, rows


def verify(dest_dir: Path, expected: int) -> tuple[int, list[str]]:
    """Re-walk the collected tree and confirm it is internally consistent.

    Returns ``(n_links, problems)``. A link is sound when it is a regular
    file with ``st_nlink >= 2`` — i.e. it genuinely shares its inode with
    the delivered original rather than being a stray copy.
    """
    problems: list[str] = []
    links = sorted(dest_dir.glob(f"member.uid___A001_*/{TP_CUBE_GLOB_SUFFIX}"))

    for link in links:
        try:
            st = link.stat()
        except OSError as exc:
            problems.append(f"{link}: stat failed ({exc})")
            continue
        if not link.is_file():
            problems.append(f"{link}: not a regular file")
        elif st.st_nlink < 2:
            problems.append(f"{link}: st_nlink={st.st_nlink} (not a hard link to the original)")

    if len(links) != expected:
        problems.append(f"link count {len(links)} != expected {expected}")

    return len(links), problems
