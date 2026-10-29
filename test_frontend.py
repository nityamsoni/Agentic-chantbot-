import ast
import json
import logging
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from streamlit.testing.v1 import AppTest


class FakeGraph:
    def __init__(self) -> None:
        self.messages = {}
        self.fail = False
        self.reads = 0

    def get_state(self, config):
        self.reads += 1
        return SimpleNamespace(values={"messages": self.messages.get(config["configurable"]["thread_id"], [])})

    def stream(self, state, config, stream_mode):
        assert stream_mode == ["messages", "updates"]
        thread_id = config["configurable"]["thread_id"]
        messages = self.messages.setdefault(thread_id, [])
        messages.extend(state["messages"])
        if self.fail:
            raise TimeoutError("Simulated timeout")
        request = AIMessage(content="", tool_calls=[{"name": "rag_tool", "args": {"query": "skills"}, "id": "call1"}])
        tool = ToolMessage(name="rag_tool", tool_call_id="call1", content=json.dumps({
            "source_file": "resume.pdf", "context": ["Python and LangGraph"], "metadata": [{"page": 0}]
        }))
        yield "updates", {"chat_node": {"messages": [request]}}
        yield "updates", {"tools": {"messages": [tool]}}
        for text in ("Strong ", "Python skills."):
            yield "messages", (AIMessageChunk(content=text), {"langgraph_node": "chat_node"})
        final = AIMessage(content="Strong Python skills.")
        messages.extend([request, tool, final])
        yield "updates", {"chat_node": {"messages": [final]}}


class FrontendTests(unittest.TestCase):
    def setUp(self) -> None:
        st.cache_resource.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.graph = FakeGraph()
        self.backend = SimpleNamespace(chatbot=self.graph, retrive_threads=lambda: list(self.graph.messages))
        self.source = Path(__file__).with_name("streamlit_frontend_resumechat_feature.py").read_text(encoding="utf-8")
        self.source = self.source.replace('AUTH_DB_PATH = "auth.db"',
                                          "AUTH_DB_PATH = " + repr(str(Path(self.temp.name) / "auth.db")))
        previous = sys.modules.get("langchain_backend")
        sys.modules["langchain_backend"] = self.backend

        def restore_backend() -> None:
            if previous is None:
                sys.modules.pop("langchain_backend", None)
            else:
                sys.modules["langchain_backend"] = previous

        self.addCleanup(restore_backend)
        self.app = AppTest.from_string(self.source, default_timeout=30)

    def assert_clean(self) -> None:
        self.assertFalse(self.app.exception, [e.message for e in self.app.exception])

    def login(self) -> None:
        self.app.session_state["auth_user"] = {"id": 1, "name": "Test User"}
        self.app.run()
        self.assert_clean()

    def test_auth_forms_and_registration(self) -> None:
        self.app.run()
        self.assert_clean()
        self.app.segmented_control[0].set_value("Register").run()
        for field, value in zip(self.app.text_input, ("Test User", "test@example.com", "123456789", "testing-password")):
            field.set_value(value)
        self.app.button[0].click().run()
        self.assert_clean()
        self.assertTrue(self.app.success)
        self.app.segmented_control[0].set_value("Login").run()
        self.app.text_input[0].set_value("test@example.com")
        self.app.text_input[1].set_value("testing-password")
        self.app.button[0].click().run()
        self.assert_clean()
        self.assertEqual(self.app.session_state["auth_user"]["email"], "test@example.com")

    def test_stream_sources_restore_and_clear(self) -> None:
        self.login()
        self.app.chat_input[0].set_value("Review skills").run()
        self.assert_clean()
        answer = self.app.session_state["message_history"][-1]
        self.assertEqual(answer["content"], "Strong Python skills.")
        self.assertEqual(answer["sources"][0]["page"], 1)
        self.assertIsNotNone(answer["timing"]["first_token"])
        original_thread = self.app.session_state["thread_id"]
        self.app.run()
        reads = self.graph.reads
        self.app.run()
        self.assertEqual(self.graph.reads, reads)
        clear = next(button for button in self.app.button if button.label == "Clear current chat")
        clear.click().run()
        self.assert_clean()
        self.assertNotEqual(original_thread, self.app.session_state["thread_id"])
        self.assertEqual(self.app.session_state["message_history"], [])
        self.app.button(key=f"thread_{original_thread}").click().run()
        self.assert_clean()
        self.assertEqual(self.app.session_state["message_history"][-1]["sources"][0]["filename"], "resume.pdf")

    def test_stream_failure(self) -> None:
        self.login()
        self.graph.fail = True
        with self.assertLogs(level=logging.ERROR):
            self.app.chat_input[0].set_value("Hello").run()
        self.assert_clean()
        self.assertTrue(self.app.error)
        self.assertTrue(self.app.session_state["message_history"][-1]["error"])

    def test_checkpoint_database_error(self) -> None:
        def broken_history():
            raise sqlite3.DatabaseError("database disk image is malformed")

        self.backend.retrive_threads = broken_history
        self.app.session_state["auth_user"] = {"id": 1, "name": "Test User"}
        with self.assertLogs(level=logging.ERROR) as logs:
            self.app.run()
        self.assert_clean()
        self.assertTrue(self.app.error)
        self.assertIn("database disk image is malformed", "\n".join(logs.output))
        self.assertEqual(len(self.app.chat_input), 0)
        self.assertEqual(self.graph.messages, {})
        self.backend.retrive_threads = lambda: []
        next(button for button in self.app.button if button.label == "Retry storage").click().run()
        self.assert_clean()
        self.assertEqual(len(self.app.chat_input), 1)
        self.assertFalse(self.app.error)

    def test_explicit_checkpoint_database_path(self) -> None:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.checkpoint.sqlite import SqliteSaver
        from typing import Any, Optional

        source = Path(__file__).with_name("langchain_backend.py").read_text(encoding="utf-8")
        node = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "create_checkpointer")
        namespace = {"Any": Any, "Optional": Optional, "sqlite3": sqlite3, "os": os,
                     "MemorySaver": MemorySaver, "SqliteSaver": SqliteSaver}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<checkpoint-test>", "exec"), namespace)
        original = Path(self.temp.name) / "chatbot.db"
        damaged_content = b"damaged database fixture"
        original.write_bytes(damaged_content)
        replacement = Path(self.temp.name) / "chatbot-fresh.db"
        with patch.dict(os.environ, {"CHATBOT_DB_PATH": str(replacement)}):
            saver, connection = namespace["create_checkpointer"]()
            try:
                self.assertEqual(list(saver.list(None)), [])
                self.assertEqual(connection.execute("PRAGMA quick_check").fetchall(), [("ok",)])
            finally:
                connection.close()
        self.assertEqual(original.read_bytes(), damaged_content)
        with patch.dict(os.environ, {"CHATBOT_DB_PATH": str(original)}):
            saver, connection = namespace["create_checkpointer"]()
            try:
                with self.assertRaises(sqlite3.DatabaseError):
                    list(saver.list(None))
            finally:
                connection.close()
        self.assertEqual(original.read_bytes(), damaged_content)

    def test_thread_filtering_and_logout(self) -> None:
        self.graph.messages = {"u2:private": [HumanMessage(content="Private")],
                               "u1:mine": [HumanMessage(content="My resume")]}
        self.login()
        self.assertNotIn("u2:private", self.app.session_state["chat_thread"])
        next(button for button in self.app.button if button.label == "Sign out").click().run()
        self.assert_clean()
        self.assertNotIn("auth_user", self.app.session_state)

    def test_sources_missing_and_malformed(self) -> None:
        node = next(node for node in ast.parse(self.source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "extract_sources")
        namespace = {"ToolMessage": ToolMessage, "Any": object, "json": json}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<sources-test>", "exec"), namespace)
        extract = namespace["extract_sources"]
        for content in ("not json", "[]", '{"context": []}', '{"error": "No document indexed"}'):
            self.assertEqual(extract(ToolMessage(content=content, name="rag_tool", tool_call_id="one")), [])


if __name__ == "__main__":
    unittest.main()