from __future__ import annotations

import html
import json
import os
import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import List
from uuid import uuid4

import sys

from dotenv import load_dotenv

load_dotenv()
LANGSMITH_DEBUG = os.getenv("LANGSMITH_DEBUG", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _log_langsmith(message: str) -> None:
    if LANGSMITH_DEBUG:
        print(f"[langsmith] {message}", file=sys.stderr)

try:
    from langsmith import Client, traceable
    from langsmith.wrappers import wrap_openai
except Exception as exc:
    _log_langsmith(f"import failed: {exc}")
    Client = None

    def traceable(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

    def wrap_openai(client):
        return client


def ensure_pydantic_v1() -> None:
    try:
        import pydantic.v1 as pydantic_v1
    except Exception:
        return

    sys.modules["pydantic"] = pydantic_v1


ensure_pydantic_v1()

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

try:
    from markdown_it import MarkdownIt
except Exception:
    MarkdownIt = None
import openai
try:
    from openai import OpenAI as OpenAIClient
except Exception:  # pragma: no cover - depends on openai version
    OpenAIClient = None
import streamlit as st

assert os.getenv("OPENAI_API_KEY")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
FOLLOWUP_MODEL = os.getenv("FOLLOWUP_MODEL", "gpt-5.2")
QUERY_EMBEDDING_MODEL = os.getenv("QUERY_EMBEDDING_MODEL", "text-embedding-3-large")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
MAX_QUERY_VARIANTS = min(int(os.getenv("MAX_QUERY_VARIANTS", "4")), 4)
MAX_RESULTS_PER_QUERY = min(int(os.getenv("MAX_RESULTS_PER_QUERY", "3")), 3)
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "clinical-guideline-chat")
LANGSMITH_PROMPT = os.getenv("LANGSMITH_PROMPT", "")
LANGSMITH_MAX_CHUNK_CHARS = int(os.getenv("LANGSMITH_MAX_CHUNK_CHARS", "800"))
SESSION_MEMORY_TURNS = int(os.getenv("SESSION_MEMORY_TURNS", "6"))
SESSION_MEMORY_MESSAGE_CHARS = int(os.getenv("SESSION_MEMORY_MESSAGE_CHARS", "400"))
SESSION_MEMORY_CHARS = int(os.getenv("SESSION_MEMORY_CHARS", "1200"))
FOLLOWUP_QUESTION_COUNT = 0
ENABLE_FOLLOWUPS = False
SAMPLE_QUESTIONS: list[str] = []

BASE_SYSTEM_PROMPT = (
    "You are a clinical guideline assistant. Answer the clinician's question "
    "using only the provided guideline excerpts. Do not add knowledge beyond "
    "the excerpts. If the excerpts do not contain the answer, say you do not "
    "have enough information. Provide concise, clinically relevant answers "
    "that are actionable for clinicians. When relevant and supported by the "
    "excerpts, explicitly state treatment decisions (e.g., initiate or avoid "
    "therapies, first-line vs second-line options). If treatment decisions "
    "are not supported by the excerpts, say so. If the excerpts include "
    "details like dosing, timing, contraindications, or monitoring, include them. "
    "Cite sources minimally in square brackets like [1]."
)
ESTONIAN_RESPONSE_PROMPT = "Vasta alati eesti keeles."

if MarkdownIt:
    MARKDOWN_RENDERER = MarkdownIt("commonmark", {"html": False, "linkify": True}).enable(
        "table"
    )
else:
    MARKDOWN_RENDERER = None


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGSMITH_API_KEY", LANGSMITH_API_KEY)
    os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)
    if not (os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2")):
        os.environ["LANGSMITH_TRACING"] = "true"

LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2")
LANGSMITH_TRACING_ENABLED = bool(LANGSMITH_API_KEY) and _env_truthy(LANGSMITH_TRACING)
LANGSMITH_CLIENT = Client(api_key=LANGSMITH_API_KEY) if (Client and LANGSMITH_API_KEY) else None

if LANGSMITH_DEBUG:
    endpoint = (
        os.getenv("LANGSMITH_ENDPOINT")
        or os.getenv("LANGCHAIN_ENDPOINT")
        or "https://api.smith.langchain.com"
    )
    _log_langsmith(
        f"tracing_enabled={LANGSMITH_TRACING_ENABLED} project={LANGSMITH_PROJECT} endpoint={endpoint}"
    )


class OpenAIAdapter:
    def __init__(self) -> None:
        if OpenAIClient:
            client = OpenAIClient()
            if LANGSMITH_TRACING_ENABLED:
                client = wrap_openai(client)
            self._client = client
            self._style = "client"
        else:
            openai.api_key = os.getenv("OPENAI_API_KEY")
            self._client = openai
            self._style = "module"

    def chat_completions_create(self, **kwargs):
        if self._style == "client":
            return self._client.chat.completions.create(**kwargs)
        return self._client.ChatCompletion.create(**kwargs)

    def extract_message_content(self, response) -> str:
        try:
            if self._style == "client":
                return (response.choices[0].message.content or "").strip()
            return (response["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            return ""

    def extract_stream_delta(self, event) -> str | None:
        try:
            if self._style == "client":
                return event.choices[0].delta.content
            return event["choices"][0]["delta"].get("content")
        except Exception:
            return None


def traceable_if_enabled(name: str):
    def decorator(func):
        if not LANGSMITH_TRACING_ENABLED:
            return func
        return traceable(name=name)(func)

    return decorator


def get_openai_client() -> OpenAIAdapter:
    return OpenAIAdapter()


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _langsmith_enabled() -> bool:
    return LANGSMITH_TRACING_ENABLED and LANGSMITH_CLIENT is not None


def _langsmith_now() -> datetime:
    return datetime.now(timezone.utc)


def _serialize_history_for_langsmith(history: list[dict]) -> list[dict]:
    items: list[dict] = []
    for message in history:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        content = truncate_text(message.get("content", ""), SESSION_MEMORY_MESSAGE_CHARS)
        if content:
            items.append({"role": role, "content": content})
    return items


def _serialize_chunks_for_langsmith(chunks: List["CitationChunk"]) -> list[dict]:
    serialized: list[dict] = []
    for chunk in chunks:
        serialized.append(
            {
                "doc_id": chunk.doc_id,
                "source": chunk.source,
                "breadcrumbs": chunk.breadcrumbs,
                "distance": chunk.distance,
                "text": truncate_text(chunk.text, LANGSMITH_MAX_CHUNK_CHARS),
            }
        )
    return serialized


def _serialize_citations_for_langsmith(chunks: List["CitationChunk"]) -> list[dict]:
    serialized: list[dict] = []
    for chunk in chunks:
        serialized.append(
            {
                "doc_id": chunk.doc_id,
                "index": chunk.index,
                "display_index": chunk.display_index,
                "label": chunk.label,
            }
        )
    return serialized


def _start_langsmith_run(query: str, history: list[dict]) -> object | None:
    if not _langsmith_enabled():
        _log_langsmith("skip create_run (disabled or no client)")
        return None
    run_id = uuid4()
    inputs = {
        "query": query,
        "history": _serialize_history_for_langsmith(history),
    }
    extra = {
        "metadata": {
            "generation_model": OPENAI_MODEL,
            "embedding_model": QUERY_EMBEDDING_MODEL,
        }
    }
    try:
        LANGSMITH_CLIENT.create_run(
            id=run_id,
            name="chat_turn",
            run_type="chain",
            inputs=inputs,
            project_name=LANGSMITH_PROJECT,
            start_time=_langsmith_now(),
            extra=extra,
        )
    except Exception as exc:
        _log_langsmith(f"create_run failed: {exc}")
        return None
    return run_id


def _finish_langsmith_run(
    run_id: object | None,
    *,
    search_queries: list[str],
    chunks: List["CitationChunk"],
    response: str,
    citations: List["CitationChunk"],
) -> None:
    if not run_id or not _langsmith_enabled():
        return
    outputs = {
        "search_queries": search_queries,
        "retrieved_chunks": _serialize_chunks_for_langsmith(chunks),
        "response": response,
        "citations": _serialize_citations_for_langsmith(citations),
    }
    try:
        LANGSMITH_CLIENT.update_run(
            run_id,
            outputs=outputs,
            end_time=_langsmith_now(),
        )
    except Exception as exc:
        _log_langsmith(f"update_run failed: {exc}")
        return


def build_history(messages: list[dict], current_query: str) -> list[dict]:
    history: list[dict] = []
    for message in messages:
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        history.append({"role": role, "content": message.get("content", "")})
    if history and history[-1]["role"] == "user" and history[-1]["content"] == current_query:
        history = history[:-1]
    return history


def build_memory_snippet(history: list[dict]) -> str:
    if not history or SESSION_MEMORY_TURNS <= 0:
        return ""
    recent = history[-(SESSION_MEMORY_TURNS * 2) :]
    lines: list[str] = []
    for message in recent:
        role_label = "User" if message["role"] == "user" else "Assistant"
        content = truncate_text(message["content"], SESSION_MEMORY_MESSAGE_CHARS)
        if content:
            lines.append(f"{role_label}: {content}")
    snippet = "\n".join(lines)
    return truncate_text(snippet, SESSION_MEMORY_CHARS)


def bump_citation_token() -> None:
    st.session_state.citation_token = st.session_state.get("citation_token", 0) + 1


def build_citation_scope(token: int, group: int) -> str:
    return f"r{token}-a{group}"


def reset_chat_state() -> None:
    st.session_state.messages = []
    st.session_state.citation_group_counter = 0
    st.session_state.citation_token = 0
    st.session_state.pending_query = ""
    st.session_state.examples_dismissed = False
    st.session_state.clear_query_input = False
    if "query_input" in st.session_state:
        st.session_state.query_input = ""


def queue_query(text: str, dismiss_examples: bool = False) -> None:
    st.session_state.pending_query = text
    st.session_state.clear_query_input = True
    if dismiss_examples:
        st.session_state.examples_dismissed = True
    bump_citation_token()
    st.rerun()


def inject_chat_styles(has_messages: bool, show_examples: bool) -> None:
    sample_spacer_height = "28vh" if show_examples else "0"
    if has_messages:
        position_css = "bottom: 1.25rem; top: auto; transform: translateX(-50%);"
    else:
        position_css = "top: 50%; bottom: auto; transform: translate(-50%, -50%);"
    st.markdown(
        f"""
        <style>
        :root {{
          --sidebar-offset: 0px;
          --chat-input-width: min(820px, calc(92vw - var(--sidebar-offset)));
        }}
        :root:has(section[data-testid="stSidebar"][aria-expanded="true"]),
        :root:has(section[data-testid="stSidebar"][data-state="expanded"]),
        :root:has(section[data-testid="stSidebar"][aria-hidden="false"]) {{
          --sidebar-offset: 21rem;
        }}
        div[data-testid="stChatInput"] {{
          position: fixed;
          left: calc(50% + (var(--sidebar-offset) / 2));
          width: var(--chat-input-width);
          {position_css}
          z-index: 1000;
          transition: top 0.6s ease, bottom 0.6s ease, left 0.4s ease, width 0.4s ease;
        }}
        @media (max-width: 900px) {{
          :root {{
            --sidebar-offset: 0px;
            --chat-input-width: min(92vw, 640px);
          }}
          div[data-testid="stChatInput"] {{
            left: 50%;
          }}
        }}
        section.main .block-container {{
          padding-bottom: 9rem;
        }}
        .sample-spacer {{
          height: {sample_spacer_height};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@dataclass
class CitationChunk:
    index: int
    doc_id: str
    text: str
    source: str | None
    year: str | None
    breadcrumbs: str | None
    distance: float | None = None
    display_index: int | None = None

    @property
    def label(self) -> str:
        parts = []
        if self.source:
            parts.append(self.source)
        if self.year:
            parts.append(str(self.year))
        if self.breadcrumbs:
            parts.append(self.breadcrumbs)
        return " | ".join(parts) if parts else "Clinical guideline excerpt"


def build_question_history(history: list[dict]) -> str:
    if not history or SESSION_MEMORY_TURNS <= 0:
        return ""
    questions: list[str] = []
    for message in history:
        if message.get("role") != "user":
            continue
        content = truncate_text(message.get("content", ""), SESSION_MEMORY_MESSAGE_CHARS)
        if content:
            questions.append(content)
    if not questions:
        return ""
    questions = questions[-SESSION_MEMORY_TURNS:]
    history_text = "\n".join(f"- {question}" for question in questions)
    return truncate_text(history_text, SESSION_MEMORY_CHARS)


def build_followup_prompt(query: str, answer: str, history: list[dict]) -> tuple[str, str]:
    history_text = build_question_history(history)
    system_prompt = (
        "Sa oled kliiniliste ravijuhiste assistent. Paku kuni "
        f"{FOLLOWUP_QUESTION_COUNT} lühikest ja mõistlikku "
        "jätkuküsimust, mida arst küsiks pärast antud vastust. Küsimused "
        "peavad tuginema ainult kliiniku küsimusele, vastusele ja varasemale "
        "küsimuste ajaloole ning olema sellised, mida arst mõistlikult otsiks "
        "ravijuhendist (nt ravi valik, jälgimine, kriteeriumid, vastunäidustused, "
        "järgmised sammud). Väldi üldisi või juhendiväliseid küsimusi. Ära korda "
        "algset küsimust. Vasta eesti keeles. Tagasta iga küsimus eraldi real "
        "ilma nummerduseta. Kui mõistlikke jätkuküsimusi ei ole, tagasta tühi vastus."
    )
    if history_text:
        user_content = (
            f"Varasemate küsimuste ajalugu:\n{history_text}\n\n"
            f"Kliiniline küsimus: {query}\n\nVastus: {answer}"
        )
    else:
        user_content = f"Kliiniline küsimus: {query}\n\nVastus: {answer}"
    return system_prompt, user_content


def clean_followup_lines(
    raw: str, add_question_mark: bool, max_items: int = FOLLOWUP_QUESTION_COUNT
) -> list[str]:
    if not raw:
        return []
    questions: list[str] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"^\s*[\-\*\u2022]\s*", "", cleaned)
        cleaned = re.sub(r"^\s*\d+[\).\s]+\s*", "", cleaned)
        cleaned = cleaned.strip()
        if not cleaned:
            continue
        if add_question_mark and not cleaned.endswith("?"):
            cleaned = f"{cleaned}?"
        cleaned = truncate_text(cleaned, 140)
        if cleaned not in seen:
            seen.add(cleaned)
            questions.append(cleaned)
        if max_items > 0 and len(questions) >= max_items:
            break
    return questions


def parse_followup_questions(raw: str) -> list[str]:
    return clean_followup_lines(raw, add_question_mark=True, max_items=FOLLOWUP_QUESTION_COUNT)


def format_followup_preview(raw: str) -> str:
    lines = clean_followup_lines(
        raw, add_question_mark=False, max_items=FOLLOWUP_QUESTION_COUNT
    )
    if not lines:
        cleaned = raw.strip()
        return cleaned or "_Koostan jätkuküsimusi..._"
    return "\n".join(f"- {line}" for line in lines)


@traceable_if_enabled("suggest_followup_questions")
def suggest_followup_questions(
    client: OpenAIAdapter,
    query: str,
    answer: str,
    history: list[dict],
) -> list[str]:
    if not answer.strip():
        return []
    system_prompt, user_content = build_followup_prompt(query, answer, history)
    try:
        response = client.chat_completions_create(
            model=FOLLOWUP_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
        )
    except Exception:
        return []
    raw = client.extract_message_content(response)
    if not raw:
        return []
    return parse_followup_questions(raw)


@traceable_if_enabled("stream_followup_questions")
def stream_followup_questions(
    client: OpenAIAdapter, query: str, answer: str, history: list[dict]
) -> object | None:
    if not answer.strip():
        return None
    system_prompt, user_content = build_followup_prompt(query, answer, history)
    try:
        return client.chat_completions_create(
            model=FOLLOWUP_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
            stream=True,
        )
    except Exception:
        return None


def render_question_chips(
    questions: list[str],
    key_prefix: str,
    label: str | None = None,
    dismiss_examples: bool = False,
    container: st.delta_generator.DeltaGenerator = st,
) -> None:
    if not questions:
        return
    if label:
        container.caption(label)
    for idx, question in enumerate(questions):
        if container.button(
            question, key=f"{key_prefix}-{idx}", use_container_width=True
        ):
            queue_query(question, dismiss_examples=dismiss_examples)


def _extract_prompt_text(prompt: object) -> str | None:
    if isinstance(prompt, str):
        text = prompt.strip()
        return text or None
    if isinstance(prompt, dict):
        text = prompt.get("prompt")
        if isinstance(text, str) and text.strip():
            return text.strip()
        messages = prompt.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
    for attr in ("prompt", "template", "text"):
        value = getattr(prompt, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    format_func = getattr(prompt, "format", None)
    if callable(format_func):
        try:
            formatted = format_func()
        except Exception:
            return None
        if isinstance(formatted, str) and formatted.strip():
            return formatted.strip()
    return None


def render_markdown(text: str) -> str:
    if not text:
        return ""
    if MARKDOWN_RENDERER:
        try:
            return MARKDOWN_RENDERER.render(text)
        except Exception:
            pass
    escaped = html.escape(text).replace("\n", "<br/>")
    return f"<p>{escaped}</p>"


@traceable_if_enabled("load_system_prompt")
def load_system_prompt(langsmith_client: Client | None) -> tuple[str, str]:
    if not langsmith_client or not LANGSMITH_PROMPT:
        return BASE_SYSTEM_PROMPT, "local"
    try:
        prompt = langsmith_client.pull_prompt(LANGSMITH_PROMPT)
    except Exception:
        return BASE_SYSTEM_PROMPT, "langsmith-fallback"
    extracted = _extract_prompt_text(prompt)
    if extracted:
        return extracted, f"langsmith:{LANGSMITH_PROMPT}"
    return BASE_SYSTEM_PROMPT, "langsmith-fallback"


def resolve_chroma_path(path: str) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    return str((Path(__file__).resolve().parent / path_obj).resolve())


def create_chroma_client(path: str):
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
    if hasattr(chromadb, "PersistentClient"):
        return chromadb.PersistentClient(path=path)

    try:
        from chromadb.config import Settings
    except Exception as exc:
        raise SystemExit("Failed to import chromadb Settings.") from exc

    settings = Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=path,
        anonymized_telemetry=False,
    )
    return chromadb.Client(settings)


@st.cache_resource(show_spinner=False)
def get_chroma_resources(chroma_path: str, embedding_model: str):
    resolved_path = resolve_chroma_path(chroma_path)
    client = create_chroma_client(resolved_path)
    embedding_fn = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"), model_name=embedding_model
    )
    collection_names: list[str] = []
    if hasattr(client, "_db") and hasattr(client._db, "list_collections"):
        try:
            raw_rows = client._db.list_collections()
        except Exception:
            raw_rows = []
        for row in raw_rows:
            if isinstance(row, (list, tuple)) and len(row) > 1:
                collection_names.append(str(row[1]))
    else:
        try:
            raw_collections = client.list_collections()
        except Exception:
            raw_collections = []

        for item in raw_collections:
            if isinstance(item, str):
                name = item
            elif isinstance(item, dict):
                name = item.get("name")
            else:
                name = getattr(item, "name", None)
            if name:
                collection_names.append(str(name))

    collections = []
    for name in collection_names:
        try:
            collections.append(
                client.get_collection(name=name, embedding_function=embedding_fn)
            )
        except Exception:
            try:
                collections.append(
                    client.get_or_create_collection(
                        name=name, embedding_function=embedding_fn
                    )
                )
            except Exception:
                continue

    return client, embedding_fn, collections


def _parse_query_candidates(raw: str) -> list[str]:
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
    except Exception:
        pass

    lines: list[str] = []
    for line in raw.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"^\s*[\-\*\u2022]\s*", "", cleaned)
        cleaned = re.sub(r"^\s*\d+[\).\s]+\s*", "", cleaned)
        cleaned = cleaned.strip()
        if cleaned:
            lines.append(cleaned)
    return lines


@traceable_if_enabled("generate_search_queries")
def generate_search_queries(
    client: OpenAIAdapter, query: str, history: list[dict]
) -> list[str]:
    if MAX_QUERY_VARIANTS <= 1:
        return [query]

    memory = build_memory_snippet(history)
    system_prompt = (
        "You generate search queries for retrieving relevant clinical guideline "
        "passages. Produce up to 4 concise queries that cover key terms, synonyms, "
        "and related phrasing. Return only a JSON array of strings."
    )
    if memory:
        user_content = f"Conversation context:\n{memory}\n\nQuestion:\n{query}"
    else:
        user_content = query

    try:
        response = client.chat_completions_create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
    except Exception:
        return [query]

    raw = client.extract_message_content(response)
    candidates = _parse_query_candidates(raw)
    ordered: list[str] = []
    seen: set[str] = set()
    for item in [query] + candidates:
        cleaned = " ".join(item.split())
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
        if len(ordered) >= MAX_QUERY_VARIANTS:
            break

    return ordered or [query]


@traceable_if_enabled("retrieve_chunks_for_query")
def retrieve_chunks_for_query(
    query: str,
    collections: list,
    embedding_fn: OpenAIEmbeddingFunction,
    k: int = MAX_RESULTS_PER_QUERY,
    debug: bool = False,
) -> tuple[List[CitationChunk], dict]:
    debug_info: dict = {"query": query, "collections": []}
    try:
        embedding = embedding_fn([query])[0]
        if debug:
            debug_info["embedding_dim"] = len(embedding)
    except Exception as exc:
        debug_info["error"] = str(exc)
        return [], debug_info

    candidates: list[CitationChunk] = []
    for collection in collections:
        collection_name = getattr(collection, "name", None)
        entry: dict = {"collection": collection_name}
        try:
            resp = collection.query(
                query_embeddings=[embedding],
                n_results=k,
                include=["metadatas", "documents", "distances"],
            )
        except Exception as exc:
            entry["error"] = str(exc)
            if debug:
                debug_info["collections"].append(entry)
            continue

        ids = (resp.get("ids") or [[]])[0] or []
        documents = (resp.get("documents") or [[]])[0] or []
        metadatas = (resp.get("metadatas") or [[]])[0] or []
        distances = (resp.get("distances") or [[]])[0] or []
        if debug:
            entry["ids"] = ids
            entry["distances"] = distances
            entry["documents"] = documents
            entry["metadatas"] = metadatas
            debug_info["collections"].append(entry)

        for idx, doc_id in enumerate(ids):
            meta = metadatas[idx] if idx < len(metadatas) else {}
            text = ""
            if idx < len(documents):
                text = documents[idx] or ""
            if not text:
                text = meta.get("text") or meta.get("search_text") or ""
            if not text:
                continue

            distance = distances[idx] if idx < len(distances) else None
            source = (
                meta.get("source")
                or meta.get("guideline")
                or meta.get("class")
                or meta.get("collection")
                or collection_name
            )
            year = meta.get("year")
            year_str = str(year) if year is not None else None
            breadcrumbs = meta.get("breadcrumbs")

            candidates.append(
                CitationChunk(
                    index=0,
                    doc_id=str(doc_id),
                    text=text,
                    source=source,
                    year=year_str,
                    breadcrumbs=breadcrumbs,
                    distance=distance,
                )
            )

    dedup: dict[str, CitationChunk] = {}
    for cand in candidates:
        existing = dedup.get(cand.doc_id)
        if not existing:
            dedup[cand.doc_id] = cand
            continue
        cand_dist = cand.distance if cand.distance is not None else float("inf")
        existing_dist = (
            existing.distance if existing.distance is not None else float("inf")
        )
        if cand_dist < existing_dist:
            dedup[cand.doc_id] = cand

    ordered = sorted(
        dedup.values(),
        key=lambda c: c.distance if c.distance is not None else float("inf"),
    )
    return ordered[:k], debug_info


@traceable_if_enabled("retrieve_chunks_parallel")
def retrieve_chunks_parallel(
    queries: list[str],
    debug: bool = False,
) -> tuple[List[CitationChunk], dict | None]:
    _client, embedding_fn, collections = get_chroma_resources(
        CHROMA_PATH, QUERY_EMBEDDING_MODEL
    )
    if not collections:
        payload = {
            "chroma_path": resolve_chroma_path(CHROMA_PATH),
            "collections": [],
            "queries": queries,
            "per_query": [],
            "merged": [],
            "error": "No collections found",
        }
        return [], payload if debug else None

    results_by_index: dict[int, List[CitationChunk]] = {}
    debug_by_index: dict[int, dict] = {}
    for idx, query in enumerate(queries):
        try:
            chunk_list, debug_info = retrieve_chunks_for_query(
                query,
                collections,
                embedding_fn,
                MAX_RESULTS_PER_QUERY,
                debug,
            )
        except Exception as exc:
            chunk_list = []
            debug_info = {"query": query, "error": str(exc), "collections": []}
        results_by_index[idx] = chunk_list
        debug_by_index[idx] = debug_info

    combined: list[CitationChunk] = []
    seen: set[str] = set()
    for idx in range(len(queries)):
        for chunk in results_by_index.get(idx, []):
            if chunk.doc_id in seen:
                continue
            seen.add(chunk.doc_id)
            combined.append(chunk)

    for idx, chunk in enumerate(combined, start=1):
        chunk.index = idx

    payload = None
    if debug:
        payload = {
            "chroma_path": resolve_chroma_path(CHROMA_PATH),
            "collections": [getattr(c, "name", None) for c in collections],
            "queries": queries,
            "per_query": [debug_by_index.get(i) for i in range(len(queries))],
            "merged": [
                {
                    "doc_id": chunk.doc_id,
                    "distance": chunk.distance,
                    "source": chunk.source,
                    "breadcrumbs": chunk.breadcrumbs,
                    "text": chunk.text,
                }
                for chunk in combined
            ],
        }

    return combined, payload


def build_context(chunks: List[CitationChunk]) -> str:
    if not chunks:
        return ""
    lines = []
    for chunk in chunks:
        if chunk.breadcrumbs:
            excerpt = f"{chunk.breadcrumbs}\n{chunk.text}"
        else:
            excerpt = chunk.text
        lines.append(f"[{chunk.index}] {excerpt}")
    return "\n\n".join(lines)


def link_citations(text: str, indices: set[int], scope: str) -> str:
    if not text or not indices:
        return text

    def replace_match(match: re.Match[str]) -> str:
        content = match.group(1)
        if not re.fullmatch(r"[0-9,\s\-]+", content):
            return match.group(0)

        def link_number(number_match: re.Match[str]) -> str:
            idx = int(number_match.group(0))
            if idx not in indices:
                return number_match.group(0)
            return f"<a href=\"#citation-{scope}-{idx}\">{idx}</a>"

        linked = re.sub(r"\d+", link_number, content)
        return f"[{linked}]"

    return re.sub(r"\[(.*?)\]", replace_match, text)


def extract_citation_order(text: str) -> list[int]:
    if not text:
        return []
    order: list[int] = []
    seen: set[int] = set()
    for match in re.findall(r"\[(.*?)\]", text):
        if not re.fullmatch(r"[0-9,\s\-]+", match):
            continue
        for token in re.finditer(r"\d+(?:\s*-\s*\d+)?", match):
            part = token.group(0)
            if "-" in part:
                start_str, end_str = [p.strip() for p in part.split("-", 1)]
                if start_str.isdigit() and end_str.isdigit():
                    start, end = int(start_str), int(end_str)
                    if start <= end:
                        for idx in range(start, end + 1):
                            if idx not in seen:
                                seen.add(idx)
                                order.append(idx)
                continue
            if part.isdigit():
                idx = int(part)
                if idx not in seen:
                    seen.add(idx)
                    order.append(idx)
    return order


def renumber_citations(text: str, mapping: dict[int, int]) -> str:
    if not text or not mapping:
        return text

    def replace_match(match: re.Match[str]) -> str:
        content = match.group(1)
        if not re.fullmatch(r"[0-9,\s\-]+", content):
            return match.group(0)
        numbers: list[int] = []
        for token in re.finditer(r"\d+(?:\s*-\s*\d+)?", content):
            part = token.group(0)
            if "-" in part:
                start_str, end_str = [p.strip() for p in part.split("-", 1)]
                if start_str.isdigit() and end_str.isdigit():
                    start, end = int(start_str), int(end_str)
                    if start <= end:
                        numbers.extend(range(start, end + 1))
                continue
            if part.isdigit():
                numbers.append(int(part))
        if not numbers:
            return match.group(0)
        renumbered: list[int] = []
        seen_local: set[int] = set()
        for num in numbers:
            new_num = mapping.get(num, num)
            if new_num not in seen_local:
                seen_local.add(new_num)
                renumbered.append(new_num)
        return f"[{', '.join(str(num) for num in renumbered)}]"

    return re.sub(r"\[(.*?)\]", replace_match, text)


def select_citations_in_order(
    chunks: List[CitationChunk], order: list[int]
) -> List[CitationChunk]:
    if not chunks or not order:
        return []
    chunk_map = {chunk.index: chunk for chunk in chunks}
    ordered: List[CitationChunk] = []
    for idx in order:
        chunk = chunk_map.get(idx)
        if chunk:
            ordered.append(chunk)
    return ordered


def apply_citation_display(
    response: str, chunks: List[CitationChunk]
) -> tuple[str, List[CitationChunk]]:
    citation_order = extract_citation_order(response)
    if not citation_order:
        return response, []
    available_indices = {chunk.index for chunk in chunks}
    filtered_order = [idx for idx in citation_order if idx in available_indices]
    if not filtered_order:
        return response, []
    used_chunks = select_citations_in_order(chunks, filtered_order)
    display_map = {orig: idx + 1 for idx, orig in enumerate(filtered_order)}
    for chunk in used_chunks:
        chunk.display_index = display_map.get(chunk.index)
    renumbered = renumber_citations(response, display_map)
    return renumbered, used_chunks


def build_messages(
    query: str,
    chunks: List[CitationChunk],
    history: List[dict],
    system_prompt: str,
) -> list[dict]:
    context = build_context(chunks)
    if not context:
        return []
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": ESTONIAN_RESPONSE_PROMPT},
        {
            "role": "system",
            "content": f"Guideline excerpts:\n{context}",
        },
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": query})
    return messages


@traceable_if_enabled("stream_assistant_response")
def stream_assistant_response(
    client: OpenAIAdapter,
    query: str,
    chunks: List[CitationChunk],
    history: List[dict],
    system_prompt: str,
) -> tuple[bool, str | object]:
    if not chunks:
        return (
            False,
            "Ma ei leidnud sellele küsimusele asjakohaseid ravijuhendi lõike. "
            "Palun esita täpsem kliiniline küsimus.",
        )
    messages = build_messages(query, chunks, history, system_prompt)
    stream = client.chat_completions_create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        stream=True,
    )
    return True, stream


def render_citations(
    chunks: List[CitationChunk], scope: str, container: st.delta_generator.DeltaGenerator = st
) -> None:
    if not chunks:
        return
    target = container.container()
    target.markdown(
        """
        <style>
        .citation-card {
          --ink: #0f172a;
          --muted: #475569;
          --border: #e2e8f0;
          --bg: #f8fafc;
          --accent: #0ea5e9;
          --shadow: 0 12px 30px rgba(15, 23, 42, 0.08);
          border: 1px solid var(--border);
          border-left: 4px solid var(--accent);
          border-radius: 0.75rem;
          padding: 0.9rem 1.1rem;
          margin: 0.8rem 0;
          background: linear-gradient(180deg, #ffffff 0%, var(--bg) 100%);
          box-shadow: var(--shadow);
        }
        .citation-title {
          font-weight: 600;
          text-decoration: none;
          color: var(--ink);
          display: inline-flex;
          align-items: center;
          gap: 0.4rem;
        }
        .citation-title:hover {
          color: #0369a1;
        }
        .citation-index {
          background: rgba(14, 165, 233, 0.12);
          color: #0369a1;
          padding: 0.1rem 0.4rem;
          border-radius: 999px;
          font-size: 0.85rem;
        }
        .citation-meta {
          color: var(--muted);
          font-size: 0.92rem;
        }
        .citation-body {
          display: none;
          margin-top: 0.75rem;
          color: var(--ink);
          line-height: 1.55;
          font-size: 0.98rem;
        }
        .citation-card:target .citation-body {
          display: block;
        }
        .citation-body p {
          margin: 0.45rem 0;
        }
        .citation-body ul,
        .citation-body ol {
          margin: 0.45rem 0 0.45rem 1.2rem;
        }
        .citation-body li {
          margin: 0.2rem 0;
        }
        .citation-body table {
          width: 100%;
          border-collapse: collapse;
          margin: 0.6rem 0;
          font-size: 0.95rem;
        }
        .citation-body th,
        .citation-body td {
          border: 1px solid var(--border);
          padding: 0.45rem 0.6rem;
          text-align: left;
          vertical-align: top;
        }
        .citation-body thead th {
          background: #e2e8f0;
          color: #0f172a;
          font-weight: 600;
        }
        .citation-body code {
          background: #e2e8f0;
          padding: 0.1rem 0.25rem;
          border-radius: 0.3rem;
          font-size: 0.92em;
        }
        .citation-body pre {
          background: #0f172a;
          color: #f8fafc;
          padding: 0.75rem;
          border-radius: 0.6rem;
          overflow-x: auto;
        }
        .citation-body a {
          color: #0284c7;
          text-decoration: underline;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    target.markdown("### Viited")
    for chunk in chunks:
        display_index = chunk.display_index or chunk.index
        label = html.escape(chunk.label)
        body = render_markdown(chunk.text)
        target.markdown(
            f"""
            <div id="citation-{scope}-{display_index}" class="citation-card">
              <a class="citation-title" href="#citation-{scope}-{display_index}">
                <span class="citation-index">[{display_index}]</span>
                <span class="citation-meta">{label}</span>
              </a>
              <div class="citation-body">{body}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title="Clinical Guideline Chat", page_icon="🩺")
    st.title("🩺 Clinical Guideline Chat")
    st.caption("Ask clinical questions and receive guideline-based answers with citations.")
    st.caption("Click a citation like [1] to jump to the full excerpt.")
    with st.sidebar:
        if st.button("Uus vestlus", type="primary"):
            reset_chat_state()
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "citation_token" not in st.session_state:
        st.session_state.citation_token = 0
    if "citation_group_counter" not in st.session_state:
        st.session_state.citation_group_counter = 0
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = ""
    if "examples_dismissed" not in st.session_state:
        st.session_state.examples_dismissed = False
    if "clear_query_input" not in st.session_state:
        st.session_state.clear_query_input = False

    has_messages = bool(st.session_state.messages)
    ui_has_messages = has_messages or bool(st.session_state.pending_query)
    show_examples = (not ui_has_messages) and (not st.session_state.examples_dismissed) and bool(SAMPLE_QUESTIONS)
    inject_chat_styles(ui_has_messages, show_examples=show_examples)

    assistant_counter = 0
    render_token = st.session_state.citation_token
    for message in st.session_state.messages:
        role = message["role"]
        if role == "assistant":
            assistant_counter += 1
            citations = message.get("citations") or []
            group = message.get("citation_group", assistant_counter)
            with st.chat_message("assistant"):
                if citations:
                    scope = build_citation_scope(render_token, group)
                    indices = set(range(1, len(citations) + 1))
                    content = link_citations(message["content"], indices, scope)
                else:
                    content = message["content"]
                st.markdown(content, unsafe_allow_html=True)
            if citations:
                render_citations(citations, scope)
            followups = []
        else:
            with st.chat_message(role):
                st.markdown(message["content"])

    if st.session_state.clear_query_input:
        st.session_state.query_input = ""
        st.session_state.clear_query_input = False
    query = st.chat_input(
        "Küsi meditsiinialane küsimus...",
        key="query_input",
        on_submit=bump_citation_token,
    )
    if not query and st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = ""
    if query and not has_messages:
        inject_chat_styles(True, show_examples=False)
    if show_examples and not query:
        st.markdown("<div class=\"sample-spacer\"></div>", unsafe_allow_html=True)
        render_question_chips(
            SAMPLE_QUESTIONS,
            key_prefix="sample",
            label="Näidisküsimused",
            dismiss_examples=True,
        )
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        client = get_openai_client()
        langsmith_run_id = None
        status_container = st.empty()
        with status_container.status("Loon vastust", expanded=True) as status:
            status.write("(1) Koostan otsingupäringud")
            history = build_history(st.session_state.messages, query)
            langsmith_run_id = _start_langsmith_run(query, history)
            search_queries = generate_search_queries(client, query, history)
            status.write("(2) Teen andmebaasi päringut")
            chunks, _ = retrieve_chunks_parallel(search_queries, debug=False)
            status.write("(3) Loon vastust")
            system_prompt, _prompt_source = load_system_prompt(LANGSMITH_CLIENT)
            is_stream, result = stream_assistant_response(
                client, query, chunks, history, system_prompt
            )
            status.update(state="complete")

        response = ""
        used_chunks: List[CitationChunk] = []
        assistant_text = ""
        followups: list[str] = []
        citation_group = st.session_state.citation_group_counter + 1
        scope = build_citation_scope(render_token, citation_group)
        if is_stream:
            status_container.empty()

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
        citations_placeholder = st.empty()

        if not is_stream:
            response = result
            assistant_text, used_chunks = apply_citation_display(response, chunks)
            if used_chunks:
                indices = set(range(1, len(used_chunks) + 1))
                assistant_content = link_citations(assistant_text, indices, scope)
            else:
                assistant_content = assistant_text
            answer_placeholder.markdown(assistant_content, unsafe_allow_html=True)
            if used_chunks:
                citations_placeholder.empty()
                render_citations(used_chunks, scope, citations_placeholder)
        else:
            previous_order: list[int] = []
            for event in result:
                delta = client.extract_stream_delta(event)
                if delta:
                    response += delta
                    answer_placeholder.markdown(response, unsafe_allow_html=True)
                    order = extract_citation_order(response)
                    if order != previous_order:
                        assistant_text, used_chunks = apply_citation_display(response, chunks)
                        if used_chunks:
                            citations_placeholder.empty()
                            render_citations(used_chunks, scope, citations_placeholder)
                        previous_order = order
            assistant_text, used_chunks = apply_citation_display(response, chunks)
            if used_chunks:
                citations_placeholder.empty()
                render_citations(used_chunks, scope, citations_placeholder)
            if used_chunks:
                indices = set(range(1, len(used_chunks) + 1))
                assistant_content = link_citations(assistant_text, indices, scope)
            else:
                assistant_content = assistant_text
            answer_placeholder.markdown(assistant_content, unsafe_allow_html=True)

        final_response = assistant_text or response
        _finish_langsmith_run(
            langsmith_run_id,
            search_queries=search_queries,
            chunks=chunks,
            response=final_response,
            citations=used_chunks,
        )

        st.session_state.citation_group_counter = citation_group
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_text,
                "citations": used_chunks,
                "citation_group": citation_group,
                "followups": followups,
            }
        )


if __name__ == "__main__":
    main()
