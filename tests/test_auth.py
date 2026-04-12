"""Tests for panta_rei.auth — credential resolution."""

import os

import pytest

from panta_rei.auth import (
    resolve_alma_creds,
    resolve_github_token,
    read_systemd_credential,
)


class TestResolveAlmaCreds:

    def test_cli_username_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("ALMA_USERNAME", "env_user")
        username, _ = resolve_alma_creds("cli_user")
        assert username == "cli_user"

    def test_env_username_fallback(self, monkeypatch):
        monkeypatch.setenv("ALMA_USERNAME", "env_user")
        username, _ = resolve_alma_creds(None)
        assert username == "env_user"

    def test_no_username(self, monkeypatch):
        monkeypatch.delenv("ALMA_USERNAME", raising=False)
        username, _ = resolve_alma_creds(None)
        assert username is None

    def test_env_password(self, monkeypatch):
        monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
        monkeypatch.setenv("ALMA_PASSWORD", "secret123")
        _, password = resolve_alma_creds(None)
        assert password == "secret123"

    def test_systemd_credential_takes_precedence(self, tmp_path, monkeypatch):
        cred_file = tmp_path / "svc.alma"
        cred_file.write_text("systemd_password\n")
        monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))
        monkeypatch.setenv("ALMA_PASSWORD", "env_password")
        _, password = resolve_alma_creds(None)
        assert password == "systemd_password"

    def test_no_password(self, monkeypatch):
        monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
        monkeypatch.delenv("ALMA_PASSWORD", raising=False)
        _, password = resolve_alma_creds(None)
        assert password is None


class TestResolveGithubToken:

    def test_env_token(self, monkeypatch):
        monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "gh_token_123")
        assert resolve_github_token() == "gh_token_123"

    def test_systemd_credential(self, tmp_path, monkeypatch):
        cred_file = tmp_path / "github.token"
        cred_file.write_text("systemd_gh_token\n")
        monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        assert resolve_github_token() == "systemd_gh_token"

    def test_systemd_takes_precedence(self, tmp_path, monkeypatch):
        cred_file = tmp_path / "github.token"
        cred_file.write_text("systemd_token\n")
        monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))
        monkeypatch.setenv("GITHUB_TOKEN", "env_token")
        assert resolve_github_token() == "systemd_token"

    def test_no_token(self, monkeypatch):
        monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        assert resolve_github_token() is None


class TestReadSystemdCredential:

    def test_reads_file(self, tmp_path, monkeypatch):
        cred_file = tmp_path / "test.cred"
        cred_file.write_text("  credential_value  \n")
        monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))
        assert read_systemd_credential("test.cred") == "credential_value"

    def test_no_credentials_directory(self, monkeypatch):
        monkeypatch.delenv("CREDENTIALS_DIRECTORY", raising=False)
        assert read_systemd_credential("test.cred") is None

    def test_missing_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CREDENTIALS_DIRECTORY", str(tmp_path))
        assert read_systemd_credential("nonexistent") is None
