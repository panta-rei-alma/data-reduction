"""Centralized configuration for the Panta Rei pipeline.

All paths, constants, and defaults. Loads from .env file with
environment variable overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from panta_rei.core.errors import ConfigError


def _load_env_file(env_path: Path) -> dict[str, str]:
    """Load KEY=value pairs from a .env file."""
    env_vars: dict[str, str] = {}
    if not env_path.exists():
        return env_vars
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                if value and value[0] in ('"', "'") and value[-1] == value[0]:
                    value = value[1:-1]
                env_vars[key] = value
    return env_vars


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline configuration loaded from .env and environment variables.

    Attributes with defaults can be overridden via .env or env vars.
    Derived paths are computed in __post_init__.
    """

    # From .env (required)
    panta_rei_base: Path

    # From .env (optional with defaults)
    project_code: str = "2025.1.00383.L"
    casa_path: Optional[Path] = None
    weblog_dir: Path = Path("/scratch/almanas/dwalker2/panta-rei/weblogs")
    cron_log_dir: Optional[Path] = None
    python_env: Optional[Path] = None

    # Constants
    gh_owner: str = "panta-rei-alma"
    gh_repo: str = "data-reduction"
    gh_project_number: int = 1
    alma_servers: tuple[str, ...] = (
        "https://almascience.nrao.edu",
        "https://almascience.eso.org",
        "https://almascience.nao.ac.jp",
    )
    url_mappings: dict[str, str] = field(default_factory=lambda: {
        "/scratch/almanas": "https://www.alma.ac.uk/nas",
    })

    # From .env (optional — imaging-specific)
    imaging_db: Optional[Path] = None

    # Derived (set in __post_init__)
    project_dir: Path = field(init=False)
    data_dir: Path = field(init=False)
    state_db_path: Path = field(init=False)
    imaging_db_path: Path = field(init=False)
    targets_csv_path: Path = field(init=False)
    casa_cmd: Optional[str] = field(init=False)

    def __post_init__(self) -> None:
        # frozen=True requires object.__setattr__ for post-init
        object.__setattr__(
            self, "project_dir", self.panta_rei_base / self.project_code
        )
        object.__setattr__(
            self, "data_dir", self.project_dir / self.project_code
        )
        object.__setattr__(
            self, "state_db_path",
            self.project_dir / "alma_retrieval_state.sqlite3",
        )
        object.__setattr__(
            self, "imaging_db_path",
            self.imaging_db if self.imaging_db else self.project_dir / "imaging.sqlite3",
        )
        object.__setattr__(
            self, "targets_csv_path",
            self.project_dir / "targets_by_array.csv",
        )
        if self.casa_path is not None:
            casa_bin = self.casa_path / "bin" / "casa"
            object.__setattr__(
                self, "casa_cmd",
                f"{casa_bin} --nologger --nogui --pipeline",
            )
        else:
            object.__setattr__(self, "casa_cmd", None)

        if self.cron_log_dir is None:
            object.__setattr__(
                self, "cron_log_dir", self.panta_rei_base / "cron_logs"
            )

    @classmethod
    def from_env(cls, env_path: Optional[Path] = None) -> PipelineConfig:
        """Load configuration from .env file and environment variables.

        Environment variables take precedence over .env values.
        """
        if env_path is None:
            # Look for .env relative to this module's parent (the repo root)
            env_path = Path(__file__).parent.parent / ".env"

        env_vars = _load_env_file(env_path)

        def get(key: str, default: Optional[str] = None) -> Optional[str]:
            return os.environ.get(key) or env_vars.get(key) or default

        def require(key: str) -> str:
            value = get(key)
            if value is None:
                raise ConfigError(
                    f"Required configuration '{key}' not found. "
                    f"Set it in {env_path} or as an environment variable."
                )
            return value

        panta_rei_base = Path(require("PANTA_REI_BASE"))

        kwargs: dict = {
            "panta_rei_base": panta_rei_base,
            "project_code": get("PROJECT_CODE", "2025.1.00383.L"),
        }

        casa_path = get("CASA_PATH")
        if casa_path:
            kwargs["casa_path"] = Path(casa_path)

        weblog_dir = get("WEBLOG_DIR")
        if weblog_dir:
            kwargs["weblog_dir"] = Path(weblog_dir)

        cron_log_dir = get("CRON_LOG_DIR")
        if cron_log_dir:
            kwargs["cron_log_dir"] = Path(cron_log_dir)

        python_env = get("PYTHON_ENV")
        if python_env:
            kwargs["python_env"] = Path(python_env)

        imaging_db = get("IMAGING_DB")
        if imaging_db:
            kwargs["imaging_db"] = Path(imaging_db)

        gh_owner = get("GH_OWNER")
        if gh_owner:
            kwargs["gh_owner"] = gh_owner

        gh_repo = get("GH_REPO")
        if gh_repo:
            kwargs["gh_repo"] = gh_repo

        return cls(**kwargs)

    def validate(self) -> list[str]:
        """Check that configured paths exist. Returns list of issues."""
        issues: list[str] = []
        if not self.panta_rei_base.exists():
            issues.append(f"PANTA_REI_BASE does not exist: {self.panta_rei_base}")
        if self.casa_path is not None and not self.casa_path.exists():
            issues.append(f"CASA_PATH does not exist: {self.casa_path}")
        return issues

    def __str__(self) -> str:
        lines = [
            "PipelineConfig:",
            f"  panta_rei_base: {self.panta_rei_base}",
            f"  project_code: {self.project_code}",
            f"  project_dir: {self.project_dir}",
            f"  data_dir: {self.data_dir}",
            f"  state_db_path: {self.state_db_path}",
            f"  imaging_db_path: {self.imaging_db_path}",
            f"  targets_csv_path: {self.targets_csv_path}",
            f"  casa_path: {self.casa_path}",
            f"  casa_cmd: {self.casa_cmd}",
            f"  weblog_dir: {self.weblog_dir}",
            f"  cron_log_dir: {self.cron_log_dir}",
        ]
        return "\n".join(lines)
