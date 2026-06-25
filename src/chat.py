from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.rag import RAGService, get_rag_service
from src.storage import AppStorage


DEFAULT_SESSION_TITLE = "Cu\u1ed9c tr\u00f2 chuy\u1ec7n m\u1edbi"


class ChatService:
    def __init__(self, storage: AppStorage | None = None, rag: RAGService | Any | None = None) -> None:
        self.storage = storage or AppStorage()
        self.rag = rag or get_rag_service()

    @staticmethod
    def _title_from_question(question: str) -> str:
        compact = " ".join(question.split())
        return compact if len(compact) <= 80 else f"{compact[:77].rstrip()}..."

    def create_session(self, title: str | None = None) -> dict[str, Any]:
        selected_title = " ".join((title or "").split()) or DEFAULT_SESSION_TITLE
        return self.storage.create_chat_session(selected_title[:120])

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.storage.list_chat_sessions(limit)

    def get_session(self, session_id: int) -> dict[str, Any] | None:
        session = self.storage.get_chat_session(session_id)
        if session is None:
            return None
        return {**session, "messages": self.storage.list_chat_messages(session_id)}

    def delete_session(self, session_id: int) -> bool:
        return self.storage.delete_chat_session(session_id)

    def ask(self, session_id: int, question: str, **filters: Any) -> dict[str, Any] | None:
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")
        session = self.storage.get_chat_session(session_id)
        if session is None:
            return None

        messages = self.storage.list_chat_messages(session_id)
        history = [{"role": row["role"], "content": row["content"]} for row in messages]
        if not messages and session["title"] == DEFAULT_SESSION_TITLE:
            session = self.storage.update_chat_session_title(session_id, self._title_from_question(question)) or session

        user_message = self.storage.save_chat_message(session_id, "user", question)
        result = self.rag.ask(question=question, history=history, **filters)
        assistant_message = self.storage.save_chat_message(
            session_id,
            "assistant",
            str(result["answer"]),
            result=result,
        )
        session = self.storage.get_chat_session(session_id) or session
        return {
            "session": session,
            "user_message": user_message,
            "assistant_message": assistant_message,
        }


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    return ChatService()
