# 12m+7m Continuum Joint Imaging — Implementation Plan

**Date:** 2026-06-12
**Branch:** `feature/continuum-joint-imaging` (no push / PR until green-lit)
**Author:** Claude + Daniel Walker; to be reviewed by Codex pre- and post-implementation.

## 1. Goal

Add **12m+7m continuum** joint imaging to the Panta Rei pipeline, mirroring the
existing 12m+7m+TP spectral-line cube joint imaging, while introducing **zero
regressions** to the line-cube path (currently the main science priority,
working solidly).

Key differences from the line-cube path:

| Aspect | Line cube (existing) | Continuum (new) |
|---|---|---|
| Single-dish (TP) | Feathered in (12m7m**TP**) | **None** — ALMA TP is line-only. No feather step. |
| Specmode | `cube` | `mfs` (or `cont`) |
| Deconvolver | `multiscale` | **`mtmfs` always, `nterms=2`** (fixed override — see §3a) |
| Product count | thousands (per spw per line) | hundreds (per target, aggregated over spws) |
| Input MS | `*_targets_line.ms` (contsub) | `*_targets.ms` (full band, line-free freq selection) |
| Channel selection | full spw | line-free **frequency ranges** from the recovered call / `cont.dat` |
| Science product | feathered `.image.pbcor.feather` | the joint tclean `.image.pbcor` itself |

## 2. Grounding facts (verified on real data 2026-06-12)

- `*_targets.ms` (continuum, non-contsub) **and** `cont.dat` exist in every
  member `calibrated/working/` alongside `*_targets_line.ms`.
- The seed 12m aggregate-continuum `tclean` iter1 call (from the TM member's
  `casa_commands.log`, stage `s36_0`, imagename `...spwAA_BB_..cont.regcal.I.iter1`):
  ```
  vis=[EB], field='"AG344.8921+0.4602"',
  spw=['24:86.3229~86.3305GHz;86.3324~86.3801GHz,26:...,28:...,30:...,32:...,34:...'],
  datacolumn='data',
  imsize=[160,160], cell=['1.9arcsec'], phasecenter='ICRS ...',
  specmode='mfs', gridder='mosaic', mosweight=True, deconvolver='hogbom',
  restoringbeam='common', pbcor=True, weighting='briggs', robust=0.5,
  niter=20000, threshold='0.00311Jy', usemask='auto-multithresh', ...
  ```
  - **The continuum channel selection is expressed as physical frequency
    ranges in GHz, not channel indices** → portable across arrays. The 7m MS
    can reuse the same GHz ranges; only the spw *IDs* need remapping.
  - `datacolumn='data'` here (regcal). Selfcal targets → `'corrected'`. The
    existing `resolve_datacolumn()` (CORRECTED_DATA presence) handles this; we
    set `datacolumn='corrected'` and let CASA fall back per-MS, exactly as the
    line path does today.
  - Some members use `deconvolver='mtmfs'` (nterms>1) → must export `.tt0`
    (and `.alpha`) products. Plan handles nterms generically.

## 3a. Deconvolver: mtmfs always (user-confirmed 2026-06-12)

Override `deconvolver='mtmfs'`, `nterms=2` for **all** continuum images, in
**every phase and source**, regardless of what the recovered 12m call used
(some used `hogbom`, some `mtmfs`). Cheap to add and yields the spectral terms.

Implications:
- We **do not** recover/branch on `deconvolver` from the weblog — it's a FIXED
  continuum override (joins `gridder='mosaic'` etc.). Simplifies recovery.
- tclean produces `*.image.tt0`, `*.image.tt1`, `*.alpha` (+`.alpha.error`),
  each with pbcor variants. The **science product is `.image.tt0.pbcor`**
  (published to the canonical dir as `...12m7m.{freq}GHz.cont.pbcor.fits`).
  Additionally publish **`tt1`** and **`alpha`** as products the user wants
  (naming TBD: e.g. `...cont.tt1.pbcor.fits`, `...cont.alpha.fits`); residual/pb/
  mask aux follow under `aux/` (tt0 variants).
- `reffreq`: let tclean default it (center of selected freqs) unless a recovered
  value is present; for combined (Phase 2) the wide band gives a meaningful α.
- `tclean_cont.py` is written mtmfs-aware from the start (tt0/tt1/alpha export).

## 3. Two spectral setups — scope decision (NEEDS USER CONFIRMATION)

Each target has **two spectral setups** (`Line group` = `N2H+`, `HCO+`), which
are **separate SBs / GOUSs**. Three possible products:

1. **Per-setup continuum** — one continuum image per (source, setup). ~2×259.
   Clean mirror of the existing within-GOUS architecture. Recovers each setup's
   own continuum call. **Lowest risk.**
2. **Combined continuum** — one image per source spanning *all* spws of *both*
   setups. ~259. Requires **cross-GOUS aggregation by source** (union two MOUS
   groups' MSes, merge/choose imaging geometry, widen reffreq → likely
   `mtmfs nterms=2`). **Higher complexity** (new cross-GOUS grouping; the
   existing pipeline only ever groups within one GOUS).
3. **Both** (user's instinct).

**Recommendation: build in two phases — Phase 1 delivers per-setup (option 1),
Phase 2 adds combined (option 2), giving "both".** Per-setup is the low-risk
foundation that all the new machinery (recovery, MS discovery, runner, CASA
script, dispatch threading) is validated on; combined is then a
cross-GOUS aggregation layer on top, with no new CASA/runner code. This lets us
test and de-risk incrementally and ship per-setup value early.

## 4. Architecture — what we add vs. reuse

**Reuse unchanged:** the entire dispatch/distributed machinery
(`dispatch.py`, `remote_worker.py`, `staging.py`, tokens, `/raid` cache,
launcher scripts, `unit_selection.py` selection loop), `resolve_datacolumn`,
the trusted-preflight scaffolding, the DB/queries pattern, the CLI shape.

**New / extended, all gated behind a `product_type` discriminator defaulting to
`'line'` so the line path is byte-for-byte unchanged:**

### 4.1 DB (`panta_rei/db/schema.py`, queries, migration)
- Add `product_type TEXT NOT NULL DEFAULT 'line'` to `imaging_params` (and carry
  through to `imaging_runs`). Extend the table-level unique key to
  `(gous_uid, source_name, line_group, spw_id, product_type)`.
- **Unit identity (Codex #4):** continuum rows use `spw_id='cont'` (a stable
  sentinel) — the full GHz-range selection lives in the params JSON, **never** in
  the key. (Per-setup has exactly one aggregate continuum product per
  (gous, source, setup), so `'cont'` is unique.)
- **Migration must be a careful SQLite table rebuild (Codex #3):** the unique key
  is table-level (`schema.py:85`) and `ON CONFLICT` upsert depends on it
  (`models.py:456`).
  - **Preserve `imaging_params.id` exactly** (INSERT…SELECT with explicit `id`) —
    `imaging_runs.params_id` is an FK onto it.
  - Drop & recreate `idx_ip_key` with the new column list (`CREATE INDEX IF NOT
    EXISTS` will NOT update an existing index's columns).
  - Pre-check for duplicate rows where `line_group IS NULL` before adding the
    constraint (SQLite `UNIQUE` permits multiple NULLs → rebuild could surface
    collisions). Abort with a clear message if found.
  - Wrap in the existing transactional-DDL migration harness (`migrate_db`
    `BEGIN…COMMIT`/`ROLLBACK`).
- `ImagingParamsQueries.upsert/fetch` gain a `product_type` (default `'line'`).
  **Every fetch/selection path must filter by it (Codex #2)** — see §4.11.

### 4.2 Recovery (`panta_rei/imaging/recovery.py`)
- New `filter_continuum_iter1_calls()` (sibling of `filter_cube_iter1_calls`):
  match `specmode in {mfs, cont}` **and** imagename containing `.cont.` and
  `.iter1`, `restoration=True`, `pbcor=True`.
- **Preserve full continuum fidelity (Codex should-fix #3):** keep the **complete
  spw frequency-range selection string** (the line path collapses spw to its
  first id — continuum must keep the GHz ranges verbatim), plus `mosweight`,
  `pblimit`, `perchanweightdensity`, `weighting`/`robust`, all masking params
  (`usemask`, `sidelobethreshold`, `noisethreshold`, …), `threshold`,
  `reffreq` and `nterms` if present. We override `deconvolver`/`nterms` to mtmfs/2
  (§3a), but still record the recovered values for provenance. Confirm the parser
  (not just the fallback regex at `recovery.py:318`) actually extracts these.
- `recover_continuum_params_for_mous()` wrapper.

### 4.3 Workflow recovery step (`panta_rei/workflows/imaging.py`)
- A continuum recovery pass that writes per-(gous, source, setup) continuum
  param JSON sidecars and upserts `imaging_params` rows with
  `product_type='cont'`.

### 4.4 Matching (`panta_rei/imaging/matching.py`)
- `find_continuum_ms()` → `*_targets.ms` (exclude `*_targets_line.ms`).
- Continuum unit builder (separate from the line `build_imaging_units_advisory`,
  which hard-requires TP cubes at `matching.py:631`): like the line builder but
  **no TP** (`sdimage=None`, no TP discovery/regrid/freq-match); the recovered
  spw GHz-range selection is carried on the unit; `spw_id='cont'`.
- `build_cont_output_path()` / aux variant. Naming (canonical dir, since the joint
  tclean image *is* the science product):
  `group...lp_nperetto.{source}.12m7m.{freq}GHz.cont.pbcor.fits` (= `.image.tt0.pbcor`),
  plus `...cont.tt1.pbcor.fits` and `...cont.alpha.fits`; aux (residual/pb/mask,
  tt0) under `aux/`. **`.cont.` infix** distinguishes from the line aux `.cube.`.
- **Freq label (Codex should-fix #4):** `{freq}` = min/max of the recovered
  selected GHz ranges via a new continuum helper — **not** the cube-only
  `_compute_tm_freq_range()` (used at `workflows/imaging.py:500`,
  `dispatch.py:2087`).

### 4.5 Continuum trusted preflight — per-MS spw remap (Codex BLOCKING #1)
This is the genuinely new, load-bearing logic and gets its **own** preflight path
(the line preflight at `runner.py:169` / `matching.py:771` computes one center
freq → one spw per MS and is unsuitable). Algorithm:
- Parse the recovered selection string into `{src_spw_id: [(f_lo, f_hi)GHz, …]}`.
- For each MS and each frequency range, find the covering `OBSERVE_TARGET` spw via
  `msmetadata.chanfreqs`, **handling descending channel-frequency ordering** and a
  small **edge tolerance**. 12m ids usually map 1:1; 7m ids differ for the same
  physical frequencies.
- **Merge** ranges that land in the same output spw; emit per-MS
  `spw='id:f1~f2GHz;…'`.
- **Explicit coverage thresholds → hard-fail (not silent skip):** fail the unit if
  (a) an array has zero covering spw for a required range, or (b) the recovered
  selected bandwidth retained after remap drops below a configurable fraction
  (default e.g. 0.8) of the 12m selection. A bounded "warn" is allowed only above
  threshold. Record retained-bandwidth-per-array on the unit for the dry-run
  report (§4.11).
- (Nice-to-have, Codex #1) cross-check against `cont.dat` as an audit, not as the
  source of truth.

### 4.6 Runner (`panta_rei/imaging/runner.py`)
- `build_tclean_cont_params()` — FIXED params (`gridder='mosaic'`, `pbcor=True`,
  `stokes='I'`, `restoringbeam='common'`, **`deconvolver='mtmfs'`, `nterms=2`**),
  `specmode='mfs'`, recovered geometry + thresholds + masking + `mosweight` + the
  per-MS remapped spw GHz selection, `vis = vis_tm + vis_sm`.
- **Datacolumn (Codex should-fix #1):** decide explicitly — `'corrected'` iff
  **any** selected MS has `CORRECTED_DATA`, else `'data'`; mixed sets still rely
  on CASA's documented per-MS fallback (single tclean call has one global
  `datacolumn`). Covered by a real-`*_targets.ms` test.
- `run_tclean_cont()` / `run_tclean_cont_parallel()` — mirror the feather ones
  but drive the new CASA script; **no feather**.

### 4.7 CASA script (`panta_rei/casa/tclean_cont.py`, NEW — does not touch the
  working `tclean_feather.py`)
- `tclean(**params)` (always mtmfs nterms=2). **Branch on `deconvolver=='mtmfs'`,
  not `nterms>1` (Codex should-fix #2):** mtmfs products are `.tt0`-named even at
  nterms=1. Export `.image.tt0.pbcor` (science) always; export `tt1` always (we
  fix nterms=2); export `.alpha` **only if present**. Residual/pb/mask aux from
  tt0. Resume-skip keys on `.image.tt0.pbcor`. job.json-driven, same I/O contract
  as `tclean_feather.py`.

### 4.8 Dispatch threading (`dispatch.py`, `remote_worker.py`)
- `ImagingUnit` gains `product_type: str = 'line'`; serialised into the manifest
  (`serialise_unit`) and rebuilt in `remote_worker._build_imaging_unit`.
- `remote_worker` selects runner fn + CASA script by `product_type` (default →
  existing feather path at `remote_worker.py:476`; unchanged for line).
- Output-path + success/idempotency calc in `dispatch.py` (line paths at
  `dispatch.py:2081/2087`) branch on `product_type`.

### 4.9 Boundaries that MUST branch on product_type (Codex BLOCKING #2)
The default value alone is not isolation. Audit & branch/filter at **every** seam:
DB fetch/upsert; `unit_selection.select_units_to_image` (currently fetches all
recovered rows at `unit_selection.py:89` with no discriminator); the unit builder
(line builder hard-requires TP); manifest serialise/deserialise; remote worker
runner+script choice; success/`success_exists` idempotency; output-path
calculation; the recovery workflow step. Line selection MUST return only
`product_type='line'`; continuum selection only `'cont'`.

### 4.10 CLI — thin dedicated entry point (Codex nice-to-have #3)
Add a **separate `panta-rei-continuum-dispatch`** console script that calls the
shared `run_dispatch(..., product_type='cont')`. Operationally safer than a flag
on the line CLI (no chance the line cron accidentally runs continuum), and keeps
the line CLI's surface byte-for-byte unchanged. Mirror its args. A continuum
recovery entrypoint similarly parallels the line one.

### 4.11 Dry-run remap report (Codex nice-to-have #2)
The continuum `--dry-run` prints, per unit: the recovered 12m selection, the
per-MS remapped spw selections (12m + 7m), and retained selected bandwidth per
array vs. the coverage threshold. Catches 12m→7m remap mistakes before any CASA
runtime. This is the primary human check before the first live dispatch.

### 4.12 Analysis/QA (optional, Phase 3)
- `clean_diagnostics` discovery already keys on a combo token; add the continuum
  token so QA pngs can be generated. Moments are line-only (N/A for continuum).

## 5. Phasing

- **Phase 0 (done):** ground assumptions on real data. ✅
- **Phase 1:** per-setup continuum end-to-end (DB, recovery, matching, runner,
  CASA script, dispatch threading, CLI) + unit tests. Dry-run remap report, then a
  small capped live dispatch on real data. Codex confirms this is the right scope.
- **Phase 2:** combined (cross-GOUS) continuum. **Treated as a SEPARATE design,
  not a thin aggregation layer (Codex BLOCKING #5):** staging/scheduling group by a
  single `unit.gous_uid` (`dispatch.py:3212`), output naming assumes one GOUS, and
  there is no single canonical geometry/reffreq/threshold across the two setups.
  Needs a synthetic product/group identity for the merged-source product. We'll
  write a short Phase-2 design addendum (and re-review with Codex) before coding it.
- **Phase 3 (optional):** continuum QA diagnostics.

## 6. Regression-safety guarantees
- `product_type` defaults to `'line'` everywhere; no existing call site changes
  behaviour.
- New CASA script is a separate file; `tclean_feather.py` untouched.
- New runner/matching functions are additive; line functions unchanged.
- DB migration is additive; existing rows backfill to `'line'`.
- Acceptance gate: the full existing test suite (690 passing) stays green, plus
  new continuum tests.
- **Line-isolation tests (Codex should-fix #5):** default selection returns only
  `product_type='line'`; continuum rows never require TP; line dispatch writes
  identical method/output paths as before; an old DB migrates with `imaging_params.id`
  preserved and existing successful line runs intact.
- **Datacolumn test on a real `*_targets.ms`** (corrected vs data resolution).

## 8. Codex plan review — incorporated 2026-06-12
All BLOCKING (#1 spw-remap coverage thresholds, #2 branch-at-every-boundary,
#3 careful table rebuild, #4 `spw_id='cont'` identity, #5 Phase-2-as-separate-design)
and all should-fix (#1 explicit datacolumn, #2 mtmfs-branch, #3 recovery fidelity,
#4 freq-label helper, #5 line-isolation tests) folded into §§3a,4.1–4.12,5,6.
Nice-to-haves adopted: thin continuum CLI (§4.10), dry-run remap report (§4.11),
cont.dat audit (§4.5).

## 7. Decisions (user-confirmed 2026-06-12)
1. **Scope: both, phased.** Phase 1 per-setup, Phase 2 combined (cross-GOUS). ✅
2. **Naming: `.cont.` infix** on the `12m7m` token
  (`...{source}.12m7m.{freq}GHz.cont.pbcor.fits`); aux `.cont.` under `aux/`. ✅
3. **Channel selection: reuse the recovered 12m `tclean` GHz ranges** (mirrors
  the "recover everything from the weblog" philosophy). ✅
