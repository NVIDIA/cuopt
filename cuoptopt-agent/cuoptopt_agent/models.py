"""Unified LLM client supporting Claude (Anthropic), GPT (OpenAI), and NVIDIA NIM."""

from __future__ import annotations

import os
from typing import Any

import yaml


def _load_model_config(config_path: str, model_key: str) -> dict[str, Any]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if model_key not in cfg:
        raise ValueError(
            f"Model '{model_key}' not found in {config_path}. "
            f"Available: {list(cfg.keys())}"
        )
    return cfg[model_key]


class LLMClient:
    """Thin multi-backend LLM client.

    All three providers (Anthropic, OpenAI, NVIDIA NIM) are wrapped behind a
    single ``complete()`` method that returns the assistant message text.

    NVIDIA NIM uses the OpenAI SDK with a different ``base_url``; no extra
    dependency is required.
    """

    def __init__(self, model_key: str, config_path: str) -> None:
        self.cfg = _load_model_config(config_path, model_key)
        self.provider = self.cfg["provider"]
        api_key = os.environ.get(self.cfg["api_key_env"], "")
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {self.cfg['api_key_env']} is not set. "
                f"Export it before running cuoptopt-agent."
            )

        if self.provider == "anthropic":
            import anthropic  # type: ignore[import]

            self._client = anthropic.Anthropic(api_key=api_key)
        elif self.provider in ("openai", "nvidia_nim"):
            from openai import OpenAI  # type: ignore[import]

            kwargs: dict[str, Any] = {"api_key": api_key}
            if "base_url" in self.cfg:
                kwargs["base_url"] = self.cfg["base_url"]
            self._client = OpenAI(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(self, system: str, user: str) -> str:
        """Send a system + user message and return the assistant reply."""
        if self.provider == "anthropic":
            return self._complete_anthropic(system, user)
        return self._complete_openai(system, user)

    def complete_messages(self, messages: list[dict[str, str]]) -> str:
        """Send a full message list and return the assistant reply.

        The first message with role 'system' is extracted and handled
        correctly for each provider.
        """
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        system_text = "\n\n".join(system_parts) if system_parts else ""

        if self.provider == "anthropic":
            import anthropic  # type: ignore[import]

            resp = self._client.messages.create(
                model=self.cfg["model"],
                max_tokens=self.cfg.get("max_tokens", 8192),
                system=system_text or anthropic.NOT_GIVEN,
                messages=[{"role": m["role"], "content": m["content"]} for m in non_system],
                temperature=self.cfg.get("temperature", 0.2),
            )
            return resp.content[0].text

        # OpenAI / NVIDIA NIM
        resp = self._client.chat.completions.create(
            model=self.cfg["model"],
            messages=messages,  # type: ignore[arg-type]
            max_tokens=self.cfg.get("max_tokens", 8192),
            temperature=self.cfg.get("temperature", 0.2),
        )
        return resp.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _complete_anthropic(self, system: str, user: str) -> str:
        import anthropic  # type: ignore[import]

        resp = self._client.messages.create(
            model=self.cfg["model"],
            max_tokens=self.cfg.get("max_tokens", 8192),
            system=system or anthropic.NOT_GIVEN,
            messages=[{"role": "user", "content": user}],
            temperature=self.cfg.get("temperature", 0.2),
        )
        return resp.content[0].text

    def _complete_openai(self, system: str, user: str) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        resp = self._client.chat.completions.create(
            model=self.cfg["model"],
            messages=messages,  # type: ignore[arg-type]
            max_tokens=self.cfg.get("max_tokens", 8192),
            temperature=self.cfg.get("temperature", 0.2),
        )
        return resp.choices[0].message.content or ""
