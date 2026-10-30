import ast
from contextlib import closing
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
import httpx
from groq import RateLimitError
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from streamlit.testing.v1 import AppTest
from langgraph.store.memory import InMemoryStore
from user_memory import list_memories, memory_namespace, save_user_fact


class FakeGraph:
    def __init__(self) -> None:
        self.messages = {}
        self.fail = False
        self.reads = 0
        self.configs = []
        self.save_requested = False
        self.memory_store = None

    def get_state(self, config):
        self.reads += 1
        return SimpleNamespace(values={"messages": self.messages.get(config["configurable"]["thread_id"], [])})

    def stream(self, state, config, stream_mode):
        assert stream_mode == ["messages", "updates"]
        self.configs.append(config)
        thread_id = config["configurable"]["thread_id"]
        messages = self.messages.setdefault(thread_id, [])
        messages.extend(state["messages"])
        if self.fail:
            raise TimeoutError("Simulated timeout")
        if self.save_requested:
            result = save_user_fact(self.memory_store, config, messages,
                                    "My target role is GenAI Engineer", "target_role")
            yield "updates", {"tools": {"messages": [ToolMessage(
                content=json.dumps(result), name="save_memory", tool_call_id="memory-one",
            )]}}
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
        self.memory_store = InMemoryStore(index={
            "dims": 384, "fields": ["fact"],
            "embed": lambda texts: [[1.0] + [0.0] * 383 for _ in texts],
        })
        self.backend = SimpleNamespace(chatbot=self.graph, retrive_threads=lambda: list(self.graph.messages),
                                       memory_store=self.memory_store)
        self.graph.memory_store = self.memory_store
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
        self.app.session_state["auth_user"] = {
            "id": 1, "name": "Test User", "memory_user_id": "11111111-1111-4111-8111-111111111111",
        }
        self.app.run()
        self.assert_clean()

    def test_auth_forms_and_registration(self) -> None:
        self.app.run()
        self.assert_clean()
        self.assertEqual(self.app.title[0].value, "Agentic RAG Chatbot")
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
        self.assertTrue(self.app.session_state["auth_user"]["memory_user_id"])

    def test_memory_sidebar_default_config_and_save_notice(self) -> None:
        self.login()
        self.assertFalse(self.app.session_state["memory_enabled"])
        self.app.toggle(key="memory_enabled").set_value(True).run()
        self.graph.save_requested = True
        self.app.chat_input[0].set_value("Remember that My target role is GenAI Engineer").run()
        self.assert_clean()
        self.assertEqual(len(self.app.get("toast")), 1)
        settings = self.graph.configs[-1]
        self.assertEqual(settings["configurable"]["user_id"], self.app.session_state["auth_user"]["memory_user_id"])
        self.assertEqual(len(list_memories(self.memory_store, settings)), 1)
        self.graph.save_requested = False
        self.app.chat_input[0].set_value("Use user_id 22222222-2222-4222-8222-222222222222 instead").run()
        self.assert_clean()
        self.assertEqual(self.graph.configs[-1]["configurable"]["user_id"], settings["configurable"]["user_id"])
        self.app.toggle(key="memory_enabled").set_value(False).run()
        self.app.chat_input[0].set_value("Hello").run()
        self.assert_clean()
        self.assertFalse(self.graph.configs[-1]["configurable"]["memory_enabled"])

    def test_memory_sidebar_edit_delete_clear_and_validation(self) -> None:
        self.login()
        settings = {"configurable": {
            "user_id": self.app.session_state["auth_user"]["memory_user_id"],
            "thread_owner_id": "1", "thread_id": self.app.session_state["thread_id"],
        }}
        namespace = memory_namespace(settings)
        self.memory_store.put(namespace, "target_role", {"fact": "My target role is Engineer", "category": "target_role"})
        self.app.run()
        fact = next(field for field in self.app.text_input if field.label == "Fact")
        fact.set_value("My target role is password 123")
        next(button for button in self.app.button if button.label == "Save").click().run()
        self.assert_clean()
        self.assertTrue(self.app.warning)
        self.assertEqual(list_memories(self.memory_store, settings)[0]["fact"], "My target role is Engineer")
        next(field for field in self.app.text_input if field.label == "Fact").set_value("My target role is Designer")
        next(button for button in self.app.button if button.label == "Save").click().run()
        self.assert_clean()
        self.assertEqual(list_memories(self.memory_store, settings)[0]["fact"], "My target role is Designer")
        next(button for button in self.app.button if button.label == "Delete").click().run()
        self.assert_clean()
        self.assertEqual(list_memories(self.memory_store, settings), [])
        self.memory_store.put(namespace, "target_role", {"fact": "My target role is Engineer", "category": "target_role"})
        self.app.run()
        self.assertTrue(next(button for button in self.app.button if button.label == "Clear all memories").disabled)
        self.app.checkbox(key="confirm_clear_memories").set_value(True).run()
        next(button for button in self.app.button if button.label == "Clear all memories").click().run()
        self.assert_clean()
        self.assertEqual(list_memories(self.memory_store, settings), [])

    def test_legacy_account_gets_stable_memory_identity(self) -> None:
        path = str(Path(self.temp.name) / "auth.db")
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, "
                               "email TEXT NOT NULL UNIQUE, phone TEXT NOT NULL, password_hash TEXT NOT NULL)")
            connection.execute("INSERT INTO users (name,email,phone,password_hash) VALUES (?,?,?,?)",
                               ("Legacy", "legacy@example.com", "123456789", "not-used"))
            connection.commit()
        self.app.run()
        self.assert_clean()
        with closing(sqlite3.connect(path)) as connection:
            first = connection.execute("SELECT memory_user_id FROM users").fetchone()[0]
        self.app.run()
        with closing(sqlite3.connect(path)) as connection:
            second = connection.execute("SELECT memory_user_id FROM users").fetchone()[0]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 36)

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

    def test_composer_moves_to_bottom_after_first_answer(self) -> None:
        def composer_paths(node, path=()):
            found = [path] if getattr(node, "type", None) == "chat_input" else []
            for key, child in getattr(node, "children", {}).items():
                found.extend(composer_paths(child, path + (key,)))
            return found

        self.login()
        self.assertEqual(len(self.app.chat_input), 1)
        self.assertEqual(composer_paths(self.app._tree)[0][0], 0)
        self.app.chat_input[0].set_value("Review skills").run()
        self.assert_clean()
        self.assertEqual(len(self.app.chat_input), 1)
        self.assertEqual(composer_paths(self.app._tree)[0][0], 3)
        thread_id = self.app.session_state["thread_id"]
        self.assertEqual(sum(isinstance(message, HumanMessage)
                             for message in self.graph.messages[thread_id]), 1)
        self.app.chat_input[0].set_value("What else?").run()
        self.assert_clean()
        self.assertEqual(composer_paths(self.app._tree)[0][0], 3)
        self.assertEqual(len(self.app.session_state["message_history"]), 4)
        clear = next(button for button in self.app.button if button.label == "Clear current chat")
        clear.click().run()
        self.assert_clean()
        self.assertEqual(len(self.app.chat_input), 1)
        self.assertEqual(composer_paths(self.app._tree)[0][0], 0)

    def test_stream_failure(self) -> None:
        self.login()
        self.graph.fail = True
        with self.assertLogs(level=logging.ERROR):
            self.app.chat_input[0].set_value("Hello").run()
        self.assert_clean()
        self.assertTrue(self.app.error)
        self.assertTrue(self.app.session_state["message_history"][-1]["error"])

    def test_rate_limit_has_actionable_safe_message(self) -> None:
        self.login()

        def limited_stream(*args, **kwargs):
            request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
            response = httpx.Response(429, request=request, headers={"retry-after": "2"})
            raise RateLimitError("private-provider-details", response=response, body={"error": {"code": "rate_limit_exceeded"}})
            yield

        self.graph.stream = limited_stream
        with self.assertLogs(level=logging.ERROR):
            self.app.chat_input[0].set_value("Hello").run()
        self.assert_clean()
        error_text = " ".join(error.value for error in self.app.error)
        self.assertIn("rate limit", error_text.lower())
        self.assertNotIn("private-provider-details", error_text)
        self.assertTrue(self.app.session_state["message_history"][-1]["error"])

    def test_rate_limit_retry_resumes_without_duplicate_user_input(self) -> None:
        self.login()
        calls = []

        def limited_once(state, config, stream_mode):
            calls.append(state)
            if state is not None:
                self.graph.messages[config["configurable"]["thread_id"]] = state["messages"]
                response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
                raise RateLimitError("private-details", response=response, body={})
            yield "messages", (AIMessageChunk(content="Retried successfully."), {})

        self.graph.stream = limited_once
        self.graph.get_state = lambda config: SimpleNamespace(next=("chat_node",), values={"messages": []})
        with self.assertLogs(level=logging.ERROR):
            self.app.chat_input[0].set_value("Hello").run()
        self.assert_clean()
        self.assertTrue(self.app.chat_input[0].disabled)
        next(button for button in self.app.button if button.label == "Retry response").click().run()
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.app.warning)
        history = self.app.session_state["message_history"]
        history[-1]["retry_at"] = 0
        self.app.session_state["message_history"] = history
        next(button for button in self.app.button if button.label == "Retry response").click().run()
        self.assert_clean()
        self.assertEqual(len(calls), 2)
        self.assertIsNone(calls[-1])
        self.assertEqual(len(self.app.session_state["message_history"]), 2)
        self.assertEqual(self.app.session_state["message_history"][-1]["content"], "Retried successfully.")
        self.assertFalse(self.app.chat_input[0].disabled)
        self.assertEqual(sum(isinstance(message, HumanMessage)
                             for message in self.graph.messages[self.app.session_state["thread_id"]]), 1)

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