"""Create and select named conversations within the current chat or topic."""

from __future__ import annotations

import re
import unicodedata
from contextvars import ContextVar
from datetime import datetime
from typing import Any
from uuid import uuid4

from nanobot.agent.tools.base import Tool
from nanobot.session.manager import SessionManager


class SessionManageTool(Tool):
    """Manage named sessions without exposing another chat's history."""

    def __init__(self, manager: SessionManager):
        self._sessions = manager
        self._context: ContextVar[tuple[str, str]] = ContextVar(
            "session_manage_context", default=("", ""),
        )

    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the trusted channel and chat (or thread-scoped chat identifier)."""
        self._context.set((channel, chat_id))

    @property
    def _channel(self) -> str:
        return self._context.get()[0]

    @property
    def _chat_id(self) -> str:
        return self._context.get()[1]

    @property
    def name(self) -> str:
        return "session_manage"

    @property
    def description(self) -> str:
        return (
            "Manage sessions in the current chat or topic: create, switch, list, current, reset. "
            "Create accepts a short Unicode name or a listed full key and can activate it. "
            "Reset selects the default conversation and preserves all history. "
            "A changed selection applies to subsequent user messages."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "switch", "list", "current", "reset"],
                },
                "session_key": {
                    "type": "string", "minLength": 1, "maxLength": 240,
                    "description": "Short name or full session key from this chat's list",
                },
                "title": {
                    "type": "string", "minLength": 1, "maxLength": 120,
                    "description": "Optional title for the new session",
                },
                "seed": {
                    "type": "string", "maxLength": 4000,
                    "description": "Optional text whose first line becomes the title",
                },
                "activate": {
                    "type": "boolean", "default": True,
                    "description": "Activate the session after create",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                "allow_existing": {
                    "type": "boolean", "default": False,
                    "description": "Reuse an existing named session, preserving its history",
                },
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        action: str,
        session_key: str | None = None,
        title: str | None = None,
        seed: str | None = None,
        activate: bool = True,
        limit: int = 20,
        allow_existing: bool = False,
        **kwargs: Any,
    ) -> str:
        parameters = {
            "action": action, "activate": activate, "limit": limit,
            "allow_existing": allow_existing,
        }
        for name, value in (("session_key", session_key), ("title", title), ("seed", seed)):
            if value is not None:
                parameters[name] = value
        errors = self.validate_params(parameters)
        if errors or kwargs:
            return "Error: " + "; ".join(errors or ["Unknown session parameters"])
        try:
            base = self._sessions.get_default_session_key(self._channel, self._chat_id)
            if title is not None and (not title.strip() or any(ord(c) < 32 for c in title)):
                raise ValueError("title must be a nonempty single line")
            if action == "create":
                return self._create_session(session_key, title, seed, activate, allow_existing)
            if action == "switch":
                if session_key is None:
                    raise ValueError("session_key is required for switch")
                key = self._resolve_key(session_key)
                if not self._sessions.is_session_key_in_scope(self._channel, self._chat_id, key):
                    raise ValueError("Session does not exist or belongs to another chat")
                if key != base and not self._sessions.session_exists(key):
                    raise ValueError("Session not found")
                self._sessions.set_active_session_key(self._channel, self._chat_id, key)
                return f"Switched active session to: {key}{self._title_suffix(key)}"
            if action == "list":
                return self._list_sessions(limit)
            if action == "current":
                key = self._sessions.get_active_session_key(self._channel, self._chat_id) or base
                return f"Current active session: {key}{self._title_suffix(key)}"
            self._sessions.clear_active_session_key(self._channel, self._chat_id)
            return f"Selected default session: {base}. All session history is preserved."
        except ValueError as exc:
            return f"Error: {exc}"

    def _create_session(
        self, session_key: str | None, title: str | None, seed: str | None,
        activate: bool, allow_existing: bool,
    ) -> str:
        base = self._sessions.get_default_session_key(self._channel, self._chat_id)
        if session_key is None:
            key = f"{base}:session:{uuid4().hex}"
            while self._sessions.session_exists(key):
                key = f"{base}:session:{uuid4().hex}"
        else:
            key = self._resolve_key(session_key)
        if key == base:
            raise ValueError("Use switch or reset to select the default conversation")
        self._validate_key_length(key)
        exists = self._sessions.session_exists(key)
        if exists:
            if not allow_existing:
                raise ValueError(f"Session already exists: {key}")
            if not self._sessions.is_session_key_in_scope(self._channel, self._chat_id, key):
                raise ValueError("Existing session belongs to another chat or has no owner")
        session = self._sessions.get_or_create(key)
        session.metadata["session_scope"] = {"channel": self._channel, "chat_id": self._chat_id}
        if title is not None:
            session.metadata["title"] = title.strip()
        elif not session.metadata.get("title"):
            session.metadata["title"] = self._generate_title(seed)
        self._sessions.save(session)
        if activate:
            self._sessions.set_active_session_key(self._channel, self._chat_id, key)
        status = "Created and activated" if activate else "Created"
        return f"{status} session: {key}{self._title_suffix(key)}"

    def _list_sessions(self, limit: int) -> str:
        base = self._sessions.get_default_session_key(self._channel, self._chat_id)
        active = self._sessions.get_active_session_key(self._channel, self._chat_id) or base
        sessions = [
            info for info in self._sessions.list_sessions()
            if self._sessions.is_session_key_in_scope(self._channel, self._chat_id, info["key"])
        ]
        if not sessions:
            return "No sessions found in this chat."
        lines = []
        for index, info in enumerate(sessions[:limit], start=1):
            key = info["key"]
            mark = "*" if key == active else " "
            label = f"{index}. {mark} {key}"
            if info.get("title"):
                label += f" | {info['title']}"
            if info.get("updated_at"):
                label += f" | updated {info['updated_at']}"
            lines.append(label)
        return "Sessions:\n" + "\n".join(lines)

    def _title_suffix(self, key: str) -> str:
        title = self._sessions.get_session_title(key)
        return f" (title: {title})" if title else ""

    def _resolve_key(self, value: str) -> str:
        base = self._sessions.get_default_session_key(self._channel, self._chat_id)
        if not value.strip() or ".." in value or any(
            ord(c) < 32 or c in '<>"/\\|?*' for c in value
        ):
            raise ValueError("Invalid session key")
        if ":" in value:
            prefix = f"{base}:session:"
            if value == base:
                return value
            if not value.startswith(prefix):
                raise ValueError("Session key is outside the current chat")
            suffix = value[len(prefix):]
            if not suffix or suffix != self._slugify(suffix):
                raise ValueError("Invalid named session key")
            key = value
        else:
            key = f"{base}:session:{self._slugify(value)}"
        self._validate_key_length(key)
        return key

    @staticmethod
    def _validate_key_length(key: str) -> None:
        if len(key.encode("utf-8")) > 240:
            raise ValueError("Session key is too long for storage")

    @staticmethod
    def _generate_title(seed: str | None) -> str:
        if seed and seed.strip():
            return seed.strip().splitlines()[0].strip()[:120]
        return f"Session {datetime.now():%Y-%m-%d %H:%M}"

    @staticmethod
    def _slugify(text: str) -> str:
        text = unicodedata.normalize("NFKC", text).strip().casefold()
        if any(not (c.isalnum() or c in "-_" or c.isspace()) for c in text):
            raise ValueError("Use letters, numbers, spaces, hyphens, or underscores in a name")
        slug = re.sub(r"\s+", "-", text).strip("-_")
        if not slug:
            raise ValueError("Session name must include a letter or number")
        return slug
