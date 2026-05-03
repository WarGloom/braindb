"""
Pure-unit coverage for LLM provider profile resolution in braindb.config.

No live stack: these construct Settings() directly with _env_file=None and
monkeypatched env, so they run with `pytest` even when the test stack is down
(they request no `api` fixture). The focus is the `openai_compatible` profile
and - critically - that its env-driven base URL does NOT leak into any other
profile's resolution (the scoping guarantee from PR #7's review).
"""
import pytest

from braindb.config import Settings


pytestmark = pytest.mark.unit


def test_codex_profile_resolves_default_model():
    settings = Settings(_env_file=None, llm_profile="codex", agent_model="", openai_api_key="test-key")

    assert settings.resolved_agent_model == "openai/gpt-5.3-codex-spark"


def test_codex_profile_resolves_api_key_from_field():
    settings = Settings(_env_file=None, llm_profile="codex", openai_api_key="test-key")

    assert settings.resolved_api_key == "test-key"


def test_codex_profile_resolves_api_key_from_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    settings = Settings(_env_file=None, llm_profile="codex")

    assert settings.resolved_api_key == "env-key"


def test_agent_model_override_wins_for_codex_profile():
    settings = Settings(
        _env_file=None,
        llm_profile="codex",
        agent_model="openai/alternate-model",
        openai_api_key="test-key",
    )

    assert settings.resolved_agent_model == "openai/alternate-model"


def test_unknown_profile_error_lists_known_profiles():
    settings = Settings(_env_file=None, llm_profile="missing", agent_model="")

    with pytest.raises(ValueError, match="openai_compatible"):
        _ = settings.resolved_agent_model


def test_openai_compatible_resolves_env_values(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "openai/gpt-5-mini")
    monkeypatch.setenv("AGENT_BASE_URL", "http://localhost:4141/v1")
    monkeypatch.setenv("AGENT_API_KEY", "test-key")
    monkeypatch.setenv("AGENT_USE_RESPONSES", "true")

    s = Settings(_env_file=None, llm_profile="openai_compatible")

    assert s.resolved_agent_model == "openai/gpt-5-mini"
    assert s.resolved_base_url == "http://localhost:4141/v1"
    assert s.resolved_api_key == "test-key"
    assert s.agent_use_responses is True


def test_openai_compatible_empty_key_uses_default(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "openai/llama3.2:3b")
    monkeypatch.setenv("AGENT_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("AGENT_API_KEY", raising=False)

    s = Settings(_env_file=None, llm_profile="openai_compatible")

    assert s.resolved_base_url == "http://localhost:11434/v1"
    assert s.resolved_api_key == "ollama"


def test_openai_compatible_without_base_url_is_none(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "openai/llama3.2:3b")
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)
    monkeypatch.delenv("AGENT_API_KEY", raising=False)

    s = Settings(_env_file=None, llm_profile="openai_compatible")

    assert s.resolved_base_url is None
    assert s.resolved_api_key == "ollama"


def test_agent_base_url_does_not_leak_into_other_profiles(monkeypatch):
    monkeypatch.setenv("AGENT_BASE_URL", "http://evil-override:9999/v1")

    vllm = Settings(_env_file=None, llm_profile="vllm_workstation")
    assert vllm.resolved_base_url == "http://host.docker.internal:8002/v1"

    deepinfra = Settings(_env_file=None, llm_profile="deepinfra")
    assert deepinfra.resolved_base_url is None

    nim = Settings(_env_file=None, llm_profile="nim")
    assert nim.resolved_base_url is None


@pytest.mark.parametrize(
    ("profile", "expected_model", "expected_base_url"),
    [
        ("deepinfra", "deepinfra/google/gemma-4-31B-it", None),
        ("nim", "nvidia_nim/google/gemma-4-31b-it", None),
        ("vllm_workstation", "openai/cyankiwi/gemma-4-31B-it-AWQ-4bit",
         "http://host.docker.internal:8002/v1"),
    ],
)
def test_existing_profiles_unchanged(monkeypatch, profile, expected_model, expected_base_url):
    """Regression: the established profiles resolve exactly as before."""
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.delenv("AGENT_BASE_URL", raising=False)

    s = Settings(_env_file=None, llm_profile=profile)

    assert s.resolved_agent_model == expected_model
    assert s.resolved_base_url == expected_base_url


def test_agent_model_override_still_wins(monkeypatch):
    """AGENT_MODEL overrides the profile default for any profile."""
    monkeypatch.setenv("AGENT_MODEL", "deepinfra/some/other-model")
    s = Settings(_env_file=None, llm_profile="deepinfra")
    assert s.resolved_agent_model == "deepinfra/some/other-model"


def test_openai_compatible_profile_default_api_key():
    settings = Settings(
        _env_file=None,
        llm_profile="openai_compatible",
        agent_model="openai/gpt-5-mini",
        agent_base_url="http://localhost:4141/v1",
    )

    assert settings.resolved_agent_model == "openai/gpt-5-mini"
    assert settings.resolved_api_key == "ollama"
    assert settings.resolved_agent_base_url == "http://localhost:4141/v1"


def test_local_ollama_alias_matches_openai_compatible():
    from braindb.config import _LLM_PROFILES

    assert _LLM_PROFILES["local_ollama"] is _LLM_PROFILES["openai_compatible"]
    settings = Settings(
        _env_file=None,
        llm_profile="local_ollama",
        agent_model="openai/llama3.2:3b",
    )

    assert settings.resolved_agent_model == "openai/llama3.2:3b"
    assert settings.resolved_api_key == "ollama"
