"""Tests for panta_rei.config — PipelineConfig loading and validation."""

from pathlib import Path

import pytest

from panta_rei.config import PipelineConfig, _load_env_file
from panta_rei.core.errors import ConfigError


# ---------------------------------------------------------------------------
# _load_env_file
# ---------------------------------------------------------------------------

class TestLoadEnvFile:

    def test_load_basic(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY1=value1\nKEY2=value2\n")
        result = _load_env_file(env_file)
        assert result == {"KEY1": "value1", "KEY2": "value2"}

    def test_skip_comments_and_blanks(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nKEY=val\n")
        result = _load_env_file(env_file)
        assert result == {"KEY": "val"}

    def test_strip_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('KEY1="quoted"\nKEY2=\'single\'\n')
        result = _load_env_file(env_file)
        assert result["KEY1"] == "quoted"
        assert result["KEY2"] == "single"

    def test_value_with_equals(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=value=with=equals\n")
        result = _load_env_file(env_file)
        assert result["KEY"] == "value=with=equals"

    def test_missing_file(self, tmp_path):
        result = _load_env_file(tmp_path / "nonexistent")
        assert result == {}


# ---------------------------------------------------------------------------
# PipelineConfig construction
# ---------------------------------------------------------------------------

class TestPipelineConfig:

    def test_from_env_basic(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(f"PANTA_REI_BASE={tmp_path}\n")
        config = PipelineConfig.from_env(env_file)
        assert config.panta_rei_base == tmp_path
        assert config.project_code == "2025.1.00383.L"

    def test_derived_paths(self, tmp_path):
        config = PipelineConfig(panta_rei_base=tmp_path)
        assert config.project_dir == tmp_path / "2025.1.00383.L"
        assert config.data_dir == tmp_path / "2025.1.00383.L" / "2025.1.00383.L"
        assert config.state_db_path == tmp_path / "2025.1.00383.L" / "alma_retrieval_state.sqlite3"
        assert config.targets_csv_path == tmp_path / "2025.1.00383.L" / "targets_by_array.csv"

    def test_cron_log_dir_default(self, tmp_path):
        config = PipelineConfig(panta_rei_base=tmp_path)
        assert config.cron_log_dir == tmp_path / "cron_logs"

    def test_cron_log_dir_explicit(self, tmp_path):
        config = PipelineConfig(
            panta_rei_base=tmp_path, cron_log_dir=tmp_path / "my_logs"
        )
        assert config.cron_log_dir == tmp_path / "my_logs"

    def test_casa_cmd_when_casa_path_set(self, tmp_path):
        config = PipelineConfig(
            panta_rei_base=tmp_path, casa_path=tmp_path / "casa"
        )
        assert config.casa_cmd is not None
        assert "--nologger" in config.casa_cmd
        assert "--nogui" in config.casa_cmd
        assert "--pipeline" in config.casa_cmd

    def test_casa_cmd_none_when_no_casa_path(self, tmp_path):
        config = PipelineConfig(panta_rei_base=tmp_path)
        assert config.casa_cmd is None

    def test_env_var_overrides_env_file(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text(
            f"PANTA_REI_BASE={tmp_path}\nPROJECT_CODE=from_file\n"
        )
        monkeypatch.setenv("PROJECT_CODE", "from_env")
        config = PipelineConfig.from_env(env_file)
        assert config.project_code == "from_env"

    def test_missing_required_key_raises(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# empty\n")
        with pytest.raises(ConfigError, match="PANTA_REI_BASE"):
            PipelineConfig.from_env(env_file)

    def test_custom_project_code(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            f"PANTA_REI_BASE={tmp_path}\nPROJECT_CODE=2024.1.99999.S\n"
        )
        config = PipelineConfig.from_env(env_file)
        assert config.project_code == "2024.1.99999.S"
        assert config.project_dir == tmp_path / "2024.1.99999.S"

    def test_validate_missing_base(self, tmp_path):
        config = PipelineConfig(panta_rei_base=tmp_path / "nonexistent")
        issues = config.validate()
        assert any("PANTA_REI_BASE" in i for i in issues)

    def test_validate_existing_base(self, tmp_path):
        config = PipelineConfig(panta_rei_base=tmp_path)
        issues = config.validate()
        assert not any("PANTA_REI_BASE" in i for i in issues)

    def test_frozen(self, tmp_path):
        config = PipelineConfig(panta_rei_base=tmp_path)
        with pytest.raises(AttributeError):
            config.project_code = "changed"

    def test_str_representation(self, tmp_path):
        config = PipelineConfig(panta_rei_base=tmp_path)
        s = str(config)
        assert "PipelineConfig:" in s
        assert str(tmp_path) in s

    def test_constants_have_defaults(self, tmp_path):
        config = PipelineConfig(panta_rei_base=tmp_path)
        assert config.gh_owner == "panta-rei-alma"
        assert config.gh_repo == "data-reduction"
        assert config.gh_project_number == 1
        assert len(config.alma_servers) == 3
        assert "/scratch/almanas" in config.url_mappings
