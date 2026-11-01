"""Session management for conversation history."""

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.config.paths import get_legacy_sessions_dir
from nanobot.utils.helpers import ensure_dir, safe_filename


@dataclass
class Session:
    """
    A conversation session.

    Stores messages in JSONL format for easy reading and persistence.

    Important: Messages are append-only for LLM cache efficiency.
    The consolidation process writes summaries to MEMORY.md/HISTORY.md
    but does NOT modify the messages list or get_history() output.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0  # Number of messages already consolidated to files

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    @staticmethod
    def _find_legal_start(messages: list[dict[str, Any]]) -> int:
        """Find first index where every tool result has a matching assistant tool_call."""
        declared: set[str] = set()
        start = 0
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        declared.add(str(tc["id"]))
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid and str(tid) not in declared:
                    start = i + 1
                    declared.clear()
                    for prev in messages[start:i + 1]:
                        if prev.get("role") == "assistant":
                            for tc in prev.get("tool_calls") or []:
                                if isinstance(tc, dict) and tc.get("id"):
                                    declared.add(str(tc["id"]))
        return start

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """Return unconsolidated messages for LLM input, aligned to a legal tool-call boundary."""
        unconsolidated = self.messages[self.last_consolidated:]
        sliced = unconsolidated[-max_messages:]

        # Drop leading non-user messages to avoid starting mid-turn when possible.
        for i, message in enumerate(sliced):
            if message.get("role") == "user":
                sliced = sliced[i:]
                break

        # Some providers reject orphan tool results if the matching assistant
        # tool_calls message fell outside the fixed-size history window.
        start = self._find_legal_start(sliced)
        if start:
            sliced = sliced[start:]

        out: list[dict[str, Any]] = []
        for message in sliced:
            entry: dict[str, Any] = {"role": message["role"], "content": message.get("content", "")}
            for key in ("tool_calls", "tool_call_id", "name"):
                if key in message:
                    entry[key] = message[key]
            out.append(entry)
        return out

    def clear(self) -> None:
        """Clear all messages and reset session to initial state."""
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()

    def retain_recent_legal_suffix(self, max_messages: int) -> None:
        """Keep a legal recent suffix, mirroring get_history boundary rules."""
        if max_messages <= 0:
            self.clear()
            return
        if len(self.messages) <= max_messages:
            return

        start_idx = max(0, len(self.messages) - max_messages)

        # If the cutoff lands mid-turn, extend backward to the nearest user turn.
        while start_idx > 0 and self.messages[start_idx].get("role") != "user":
            start_idx -= 1

        retained = self.messages[start_idx:]

        # Mirror get_history(): avoid persisting orphan tool results at the front.
        start = self._find_legal_start(retained)
        if start:
            retained = retained[start:]

        dropped = len(self.messages) - len(retained)
        self.messages = retained
        self.last_consolidated = max(0, self.last_consolidated - dropped)
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.

    Sessions are stored as JSONL files in the sessions directory.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(self.workspace / "sessions")
        self.legacy_sessions_dir = get_legacy_sessions_dir()
        self._cache: dict[str, Session] = {}

    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def _get_legacy_session_path(self, key: str) -> Path:
        """Legacy global session path (~/.nanobot/sessions/)."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.legacy_sessions_dir / f"{safe_key}.jsonl"

    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.

        Args:
            key: Session key (usually channel:chat_id).

        Returns:
            The session.
        """
        if key in self._cache:
            return self._cache[key]

        session = self._load(key)
        if session is None:
            session = Session(key=key)

        self._cache[key] = session
        return session

    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)
        if not path.exists():
            legacy_path = self._get_legacy_session_path(key)
            if legacy_path.exists():
                try:
                    shutil.move(str(legacy_path), str(path))
                    logger.info("Migrated session {} from legacy path", key)
                except Exception:
                    logger.exception("Failed to migrate session {}", key)

        if not path.exists():
            return None

        try:
            messages = []
            metadata = {}
            created_at = None
            last_consolidated = 0

            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    data = json.loads(line)

                    if data.get("_type") == "metadata":
                        if data.get("key", key) != key:
                            return None  # Different keys can map to the same legacy filename.
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                        last_consolidated = data.get("last_consolidated", 0)
                    else:
                        messages.append(data)

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
                last_consolidated=last_consolidated
            )
        except Exception as e:
            logger.warning("Failed to load session {}: {}", key, e)
            return None

    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.key)
        if path.is_file():
            with open(path, encoding="utf-8") as existing:
                first_line = existing.readline().strip()
            if first_line:
                header = json.loads(first_line)
                if header.get("_type") == "metadata" and header.get("key", session.key) != session.key:
                    raise ValueError("Session key conflicts with another stored session")

        with open(path, "w", encoding="utf-8") as f:
            metadata_line = {
                "_type": "metadata",
                "key": session.key,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "last_consolidated": session.last_consolidated
            }
            f.write(json.dumps(metadata_line, ensure_ascii=False) + "\n")
            for msg in session.messages:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")

        self._cache[session.key] = session

    def invalidate(self, key: str) -> None:
        """Remove a session from the in-memory cache."""
        self._cache.pop(key, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.

        Returns:
            List of session info dicts.
        """
        sessions = {}

        for path in (*self.sessions_dir.glob("*.jsonl"), *self.legacy_sessions_dir.glob("*.jsonl")):
            try:
                # Read just the metadata line
                with open(path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            key = data.get("key") or path.stem.replace("_", ":", 1)
                            sessions.setdefault(key, {
                                "key": key,
                                "title": (data.get("metadata") or {}).get("title", ""),
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "path": str(path)
                            })
            except Exception:
                continue

        return sorted(sessions.values(), key=lambda x: x.get("updated_at") or "", reverse=True)

    def session_exists(self, key: str) -> bool:
        """Check cache and both storage locations without creating a session."""
        if key in self._cache:
            return True
        return self._get_session_path(key).is_file() or self._get_legacy_session_path(key).is_file()

    def get_existing(self, key: str) -> Session | None:
        """Read an existing session, without caching a new empty conversation."""
        if key in self._cache:
            return self._cache[key]
        session = self._load(key)
        if session is not None:
            self._cache[key] = session
        return session

    def get_session_title(self, key: str) -> str | None:
        """Get the title of a session if it has one."""
        session = self.get_existing(key)
        return session.metadata.get("title") if session is not None else None

    @staticmethod
    def get_default_session_key(channel: str, chat_id: str) -> str:
        """Build a scope key, rejecting characters unsafe for persisted filenames."""
        for part in (channel, chat_id):
            if not isinstance(part, str) or not part or part != part.strip():
                raise ValueError("channel and chat_id must be nonempty strings")
            if ".." in part or any(ord(c) < 32 or ord(c) == 127 or c in '<>"/\\|?*' for c in part):
                raise ValueError("Invalid characters in session context")
        key = f"{channel}:{chat_id}"
        if len(key.encode("utf-8")) > 240:
            raise ValueError("Session context is too long for storage")
        return key

    def is_session_key_in_scope(self, channel: str, chat_id: str, key: str) -> bool:
        """Allow the exact default or a named session with an explicit matching owner."""
        base = self.get_default_session_key(channel, chat_id)
        if not isinstance(key, str) or len(key.encode("utf-8")) > 240:
            return False
        if key == base:
            return True
        prefix = f"{base}:session:"
        if not key.startswith(prefix):
            return False
        suffix = key[len(prefix):]
        if not suffix or not all(c.isalnum() or c in "-_" for c in suffix):
            return False
        session = self.get_existing(key)
        return session is not None and session.metadata.get("session_scope") == {
            "channel": channel, "chat_id": chat_id,
        }

    def set_active_session_key(self, channel: str, chat_id: str, key: str) -> None:
        """Persist a validated override on the default session for this chat or topic."""
        base = self.get_default_session_key(channel, chat_id)
        if key == base:
            self.clear_active_session_key(channel, chat_id)
            return
        if not self.is_session_key_in_scope(channel, chat_id, key) or not (
            self._get_session_path(key).is_file() or self._get_legacy_session_path(key).is_file()
        ):
            raise ValueError("Session does not exist or belongs to another chat")
        session = self.get_or_create(base)
        session.metadata["active_session_key"] = key
        self.save(session)

    def get_active_session_key(self, channel: str, chat_id: str) -> str | None:
        """Restore a persisted override, falling back when its target is unavailable."""
        base = self.get_default_session_key(channel, chat_id)
        session = self.get_existing(base)
        if session is None:
            return None
        key = session.metadata.get("active_session_key")
        if key is None:
            return None
        if key != base and self.is_session_key_in_scope(channel, chat_id, key) and (
            self._get_session_path(key).is_file() or self._get_legacy_session_path(key).is_file()
        ):
            return key
        self.clear_active_session_key(channel, chat_id)
        return None

    def clear_active_session_key(self, channel: str, chat_id: str) -> None:
        """Return to the default conversation without deleting any session history."""
        base = self.get_default_session_key(channel, chat_id)
        session = self.get_existing(base)
        if session is not None and "active_session_key" in session.metadata:
            session.metadata.pop("active_session_key")
            self.save(session)
