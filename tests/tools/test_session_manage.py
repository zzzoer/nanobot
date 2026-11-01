"""Named sessions stay inside the current chat and survive restarts."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nanobot.agent.tools.session_manage import SessionManageTool
from nanobot.session.manager import SessionManager


class SessionManageTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.workspace = Path(self.directory.name)
        self.legacy = self.workspace / "legacy"
        self.legacy.mkdir()
        self.paths = patch("nanobot.session.manager.get_legacy_sessions_dir", return_value=self.legacy)
        self.paths.start()
        self.addCleanup(self.paths.stop)
        self.manager = SessionManager(self.workspace)
        self.tool = SessionManageTool(self.manager)
        self.tool.set_context("telegram", "123")

    def named_key(self, name):
        return f"telegram:123:session:{name}"

    async def test_all_actions_require_context(self):
        tool = SessionManageTool(self.manager)
        for action in ("create", "switch", "list", "current", "reset"):
            with self.subTest(action=action):
                self.assertIn("Error:", await tool.execute(action, session_key="research"))
        self.assertFalse(self.manager.list_sessions())

    async def test_chinese_names_are_distinct_and_titles_are_listed(self):
        await self.tool.execute("create", session_key="机器学习", title="模型实验")
        await self.tool.execute("create", session_key="申请计划", title="申请记录")

        self.assertTrue(self.manager.session_exists(self.named_key("机器学习")))
        self.assertTrue(self.manager.session_exists(self.named_key("申请计划")))
        listing = await self.tool.execute("list")
        self.assertIn("模型实验", listing)
        self.assertIn("申请记录", listing)
        self.assertIn("* " + self.named_key("申请计划"), listing)

    async def test_automatic_keys_do_not_collide(self):
        for _ in range(4):
            response = await self.tool.execute("create", activate=False)
            self.assertNotIn("Error:", response)
        named = [s for s in self.manager.list_sessions() if ":session:" in s["key"]]
        self.assertEqual(len(named), 4)
        self.assertEqual(len({s["key"] for s in named}), 4)

    async def test_switch_and_active_selection_survive_restart(self):
        await self.tool.execute("create", session_key="research", title="Research", activate=False)
        self.assertIsNone(self.manager.get_active_session_key("telegram", "123"))
        response = await self.tool.execute("switch", session_key="research")
        self.assertIn(self.named_key("research"), response)

        restarted = SessionManager(self.workspace)
        self.assertEqual(restarted.get_active_session_key("telegram", "123"), self.named_key("research"))
        tool = SessionManageTool(restarted)
        tool.set_context("telegram", "123")
        self.assertIn("Research", await tool.execute("current"))

    async def test_reset_preserves_history_and_persists_default_selection(self):
        default = self.manager.get_or_create("telegram:123")
        default.add_message("user", "default history")
        self.manager.save(default)
        await self.tool.execute("create", session_key="research")
        named = self.manager.get_or_create(self.named_key("research"))
        named.add_message("user", "named history")
        self.manager.save(named)

        response = await self.tool.execute("reset")

        self.assertIn("history", response.lower())
        restarted = SessionManager(self.workspace)
        self.assertIsNone(restarted.get_active_session_key("telegram", "123"))
        self.assertEqual(restarted.get_or_create("telegram:123").messages[0]["content"], "default history")
        self.assertEqual(restarted.get_or_create(self.named_key("research")).messages[0]["content"], "named history")
        self.assertIn("telegram:123", await self.tool.execute("current"))

    async def test_full_key_and_exact_default_can_be_selected(self):
        await self.tool.execute("create", session_key="research")
        result = await self.tool.execute("switch", session_key=self.named_key("research"))
        self.assertNotIn("Error:", result)
        result = await self.tool.execute("switch", session_key="telegram:123")
        self.assertNotIn("Error:", result)
        self.assertIsNone(self.manager.get_active_session_key("telegram", "123"))

    async def test_cross_chat_keys_cannot_be_created_or_switched(self):
        other = SessionManageTool(self.manager)
        other.set_context("telegram", "1234")
        await other.execute("create", session_key="private", title="Other chat secret")
        foreign = "telegram:1234:session:private"
        for action in ("create", "switch"):
            response = await self.tool.execute(action, session_key=foreign, allow_existing=True)
            self.assertIn("Error:", response)
        self.assertNotIn("Other chat secret", await self.tool.execute("list"))
        self.assertNotIn(foreign, await self.tool.execute("list"))
        self.assertIsNone(self.manager.get_active_session_key("telegram", "123"))

    async def test_thread_scope_does_not_include_other_topics(self):
        first = SessionManageTool(self.manager)
        first.set_context("telegram", "123:topic:1")
        await first.execute("create", session_key="notes", title="First topic")
        second = SessionManageTool(self.manager)
        second.set_context("telegram", "123:topic:2")
        self.assertNotIn("First topic", await second.execute("list"))
        self.assertIn("Error:", await second.execute("switch", session_key="telegram:123:topic:1:session:notes"))
        self.assertNotIn("First topic", await self.tool.execute("list"))

    async def test_unowned_legacy_named_key_is_not_adopted(self):
        old = self.manager.get_or_create("telegram:123:research")
        old.metadata["title"] = "Unowned old session"
        self.manager.save(old)
        self.assertIn("Error:", await self.tool.execute("switch", session_key=old.key))
        self.assertNotIn("Unowned old session", await self.tool.execute("list"))

    async def test_traversal_control_and_empty_keys_are_rejected(self):
        for key in ("../escape", "..", "a/b", "a\\b", "a\x00b", "a\nb", "", "   ", "!!!"):
            for action in ("create", "switch"):
                with self.subTest(key=key, action=action):
                    self.assertIn("Error:", await self.tool.execute(action, session_key=key))
        self.assertFalse(self.manager.list_sessions())

    async def test_existing_session_is_reused_only_when_requested(self):
        await self.tool.execute("create", session_key="research", title="Original", activate=False)
        session = self.manager.get_or_create(self.named_key("research"))
        session.add_message("user", "keep me")
        self.manager.save(session)
        self.assertIn("Error:", await self.tool.execute("create", session_key="research"))
        response = await self.tool.execute("create", session_key="research", allow_existing=True, title="Updated")
        self.assertNotIn("Error:", response)
        self.assertEqual(session.messages[0]["content"], "keep me")
        self.assertEqual(session.metadata["title"], "Updated")

    async def test_parameter_bounds_are_checked_for_direct_calls(self):
        invalid = ({"limit": 0}, {"limit": 101}, {"limit": True}, {"limit": "2"},
                   {"session_key": "x" * 241}, {"title": "x" * 121},
                   {"seed": "x" * 4001}, {"activate": "false"}, {"allow_existing": 1})
        for parameters in invalid:
            with self.subTest(parameters=parameters):
                self.assertIn("Error:", await self.tool.execute("list", **parameters))
        self.assertEqual(self.tool.parameters["properties"]["limit"]["maximum"], 100)

    async def test_reading_missing_title_does_not_create_session(self):
        key = self.named_key("missing")
        self.assertIsNone(self.manager.get_session_title(key))
        self.assertNotIn(key, self.manager._cache)
        self.assertFalse(self.manager.session_exists(key))

    async def test_legacy_existence_title_and_listing(self):
        await self.tool.execute("create", session_key="research", title="Legacy title", activate=False)
        key = self.named_key("research")
        path = self.manager._get_session_path(key)
        path.replace(self.manager._get_legacy_session_path(key))
        self.manager.invalidate(key)
        self.assertTrue(self.manager.session_exists(key))
        self.assertIn("Legacy title", await self.tool.execute("list"))
        self.assertEqual(self.manager.get_session_title(key), "Legacy title")
        self.assertNotIn("Error:", await self.tool.execute("switch", session_key="research"))

    async def test_manager_rejects_cross_scope_and_missing_active_targets(self):
        await self.tool.execute("create", session_key="research", activate=False)
        with self.assertRaises(ValueError):
            self.manager.set_active_session_key("telegram", "other", self.named_key("research"))
        with self.assertRaises(ValueError):
            self.manager.set_active_session_key("telegram", "123", self.named_key("missing"))

    async def test_deleted_active_target_falls_back_to_default(self):
        await self.tool.execute("create", session_key="research")
        self.manager._get_session_path(self.named_key("research")).unlink()
        self.assertIsNone(self.manager.get_active_session_key("telegram", "123"))
        restarted = SessionManager(self.workspace)
        self.assertIsNone(restarted.get_active_session_key("telegram", "123"))

    async def test_tampered_owner_is_rejected(self):
        await self.tool.execute("create", session_key="research", activate=False)
        session = self.manager.get_or_create(self.named_key("research"))
        session.metadata["session_scope"] = {"channel": "telegram", "chat_id": "other"}
        self.manager.save(session)
        self.assertIn("Error:", await self.tool.execute("switch", session_key=session.key))
        self.assertNotIn(session.key, await self.tool.execute("list"))

    async def test_colliding_legacy_filenames_cannot_overwrite_another_chat(self):
        first = self.manager.get_or_create("telegram:a_b")
        first.add_message("user", "private history")
        self.manager.save(first)
        colliding = self.manager.get_or_create("telegram:a:b")
        self.assertFalse(colliding.messages)

        with self.assertRaises(ValueError):
            self.manager.save(colliding)

        restarted = SessionManager(self.workspace)
        self.assertEqual(restarted.get_or_create("telegram:a_b").messages[0]["content"], "private history")

    async def test_invalid_persisted_pointer_is_cleared_without_loading_target(self):
        default = self.manager.get_or_create("telegram:123")
        default.metadata["active_session_key"] = "telegram:123:session:" + "x" * 1000
        self.manager.save(default)

        self.assertIsNone(self.manager.get_active_session_key("telegram", "123"))
        self.assertNotIn("active_session_key", default.metadata)


if __name__ == "__main__":
    unittest.main()
