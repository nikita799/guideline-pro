import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Iterable, Literal

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langsmith import Client, traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI
from pydantic import BaseModel
import weaviate
from weaviate.auth import Auth

assert os.getenv("WEAVIATE_URL")
assert os.getenv("WEAVIATE_API_KEY")
assert os.getenv("OPENAI_API_KEY")

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
FOLLOWUP_MODEL = os.getenv("FOLLOWUP_MODEL", "gpt-5.2")
COLLECTION_NAME = "Guideline"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "clinical-guideline-chat")
LANGSMITH_PROMPT = os.getenv("LANGSMITH_PROMPT", "")
SESSION_MEMORY_TURNS = int(os.getenv("SESSION_MEMORY_TURNS", "6"))
SESSION_MEMORY_MESSAGE_CHARS = int(os.getenv("SESSION_MEMORY_MESSAGE_CHARS", "400"))
SESSION_MEMORY_CHARS = int(os.getenv("SESSION_MEMORY_CHARS", "1200"))
FOLLOWUP_QUESTION_COUNT = min(int(os.getenv("FOLLOWUP_QUESTION_COUNT", "3")), 3)
REWRITE_QUERY_WAIT_SECONDS = float(os.getenv("REWRITE_QUERY_WAIT_SECONDS", "0.2"))
REWRITE_RETRIEVAL_TIMEOUT_SECONDS = float(
    os.getenv("REWRITE_RETRIEVAL_TIMEOUT_SECONDS", "0.25")
)

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
LANGSMITH_CLIENT = Client(api_key=LANGSMITH_API_KEY) if LANGSMITH_API_KEY else None

app = FastAPI(title="Clinical Guideline Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"]
    if os.getenv("ALLOW_ALL_ORIGINS", "true").lower() == "true"
    else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    query: str
    messages: list[ChatMessage] = []


@dataclass
class CitationChunk:
    index: int
    text: str
    source: str | None
    year: str | None
    breadcrumbs: str | None
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


class CitationPayload(BaseModel):
    index: int
    display_index: int
    text: str
    label: str


def traceable_if_enabled(name: str):
    def decorator(func):
        if not LANGSMITH_TRACING_ENABLED:
            return func
        return traceable(name=name)(func)

    return decorator


def get_openai_client() -> OpenAI:
    client = OpenAI()
    if LANGSMITH_TRACING_ENABLED:
        return wrap_openai(client)
    return client


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def build_history(messages: list[ChatMessage], current_query: str) -> list[dict]:
    history: list[dict] = []
    for message in messages:
        if message.role not in {"user", "assistant"}:
            continue
        history.append({"role": message.role, "content": message.content})
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


def clean_followup_lines(raw: str, add_question_mark: bool) -> list[str]:
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
        if len(questions) >= FOLLOWUP_QUESTION_COUNT:
            break
    return questions


def parse_followup_questions(raw: str) -> list[str]:
    return clean_followup_lines(raw, add_question_mark=True)


def format_followup_preview(raw: str) -> str:
    lines = clean_followup_lines(raw, add_question_mark=False)
    if not lines:
        cleaned = raw.strip()
        return cleaned or "_Koostan jätkuküsimusi..._"
    return "\n".join(f"- {line}" for line in lines)


def build_context(chunks: list[CitationChunk]) -> str:
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


def select_citations_in_order(
    chunks: list[CitationChunk], order: list[int]
) -> list[CitationChunk]:
    if not chunks or not order:
        return []
    chunk_map = {chunk.index: chunk for chunk in chunks}
    ordered: list[CitationChunk] = []
    for idx in order:
        chunk = chunk_map.get(idx)
        if chunk:
            ordered.append(chunk)
    return ordered


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


def apply_citation_display(
    response: str, chunks: list[CitationChunk]
) -> tuple[str, list[CitationChunk]]:
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


def serialize_citations(chunks: list[CitationChunk]) -> list[dict]:
    payload: list[dict] = []
    for chunk in chunks:
        display_index = chunk.display_index or chunk.index
        payload.append(
            CitationPayload(
                index=chunk.index,
                display_index=display_index,
                text=chunk.text,
                label=chunk.label,
            ).model_dump()
        )
    return payload


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


@traceable_if_enabled("retrieve_chunks")
def retrieve_chunks(query: str, k: int = 5, alpha: float = 0.8) -> list[CitationChunk]:
    with weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    ) as client:
        col = client.collections.use(COLLECTION_NAME)
        props = ["text", "search_text", "breadcrumbs", "chunk_id", "source", "year"]
        resp = col.query.hybrid(query=query, alpha=alpha, limit=k, return_properties=props)

    chunks: list[CitationChunk] = []
    for idx, obj in enumerate(resp.objects, start=1):
        props = obj.properties or {}
        chunks.append(
            CitationChunk(
                index=idx,
                text=props.get("text") or props.get("search_text") or "",
                source=props.get("source"),
                year=props.get("year"),
                breadcrumbs=props.get("breadcrumbs"),
            )
        )
    return chunks


@traceable_if_enabled("rewrite_query")
def rewrite_query(client: OpenAI, query: str, history: list[dict]) -> str:
    memory = build_memory_snippet(history)
    system_prompt = (
        "Rewrite the clinician's question into a concise search query for "
        "retrieving clinical guideline passages. Keep key clinical concepts, "
        "include relevant synonyms or alternative terms, and remove any filler. "
        "If the question is a follow-up, use the conversation context to resolve "
        "references. "
        "Return only the rewritten query."
    )
    if memory:
        user_content = f"Conversation context:\n{memory}\n\nCurrent question:\n{query}"
    else:
        user_content = query
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        rewritten = (response.choices[0].message.content or "").strip()
        return rewritten if rewritten else query
    except Exception:
        return query


def retrieve_chunks_fast(client: OpenAI, query: str, history: list[dict]) -> list[CitationChunk]:
    rewritten_query: str | None = None
    chunks: list[CitationChunk] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        rewrite_future = executor.submit(rewrite_query, client, query, history)
        retrieve_future = executor.submit(retrieve_chunks, query)
        chunks = retrieve_future.result()
        try:
            rewritten_query = rewrite_future.result(timeout=REWRITE_QUERY_WAIT_SECONDS)
        except TimeoutError:
            rewritten_query = None
        except Exception:
            rewritten_query = None
        if rewritten_query:
            rewritten_query = rewritten_query.strip()
        if (
            rewritten_query
            and rewritten_query.lower() != query.strip().lower()
            and REWRITE_RETRIEVAL_TIMEOUT_SECONDS > 0
        ):
            alt_future = executor.submit(retrieve_chunks, rewritten_query)
            try:
                alt_chunks = alt_future.result(timeout=REWRITE_RETRIEVAL_TIMEOUT_SECONDS)
            except TimeoutError:
                alt_chunks = []
            except Exception:
                alt_chunks = []
            if alt_chunks:
                return alt_chunks
    return chunks


def build_messages(
    query: str,
    chunks: list[CitationChunk],
    history: list[dict],
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
    client: OpenAI,
    query: str,
    chunks: list[CitationChunk],
    history: list[dict],
    system_prompt: str,
) -> tuple[bool, str | object]:
    if not chunks:
        return (
            False,
            "Ma ei leidnud sellele küsimusele asjakohaseid ravijuhendi lõike. "
            "Palun esita täpsem kliiniline küsimus.",
        )
    messages = build_messages(query, chunks, history, system_prompt)
    stream = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        stream=True,
    )
    return True, stream


@traceable_if_enabled("suggest_followup_questions")
def suggest_followup_questions(
    client: OpenAI,
    query: str,
    answer: str,
    history: list[dict],
) -> list[str]:
    if not answer.strip():
        return []
    system_prompt, user_content = build_followup_prompt(query, answer, history)
    try:
        response = client.chat.completions.create(
            model=FOLLOWUP_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.3,
        )
    except Exception:
        return []
    raw = (response.choices[0].message.content or "").strip()
    if not raw:
        return []
    return parse_followup_questions(raw)


@traceable_if_enabled("stream_followup_questions")
def stream_followup_questions(
    client: OpenAI, query: str, answer: str, history: list[dict]
) -> object | None:
    if not answer.strip():
        return None
    system_prompt, user_content = build_followup_prompt(query, answer, history)
    try:
        return client.chat.completions.create(
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


def sse_event(event: str, payload: dict | None = None) -> str:
    data = json.dumps(payload or {}, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


def iter_stream_events(request: ChatRequest) -> Iterable[str]:
    client = get_openai_client()
    history = build_history(request.messages, request.query)

    yield sse_event("status", {"step": 1, "text": "Teen andmebaasi päringut"})
    chunks = retrieve_chunks_fast(client, request.query, history)
    yield sse_event("status", {"step": 2, "text": "Loen vastuseid"})
    system_prompt, _prompt_source = load_system_prompt(LANGSMITH_CLIENT)
    yield sse_event("status", {"step": 3, "text": "Loon vastust"})

    is_stream, result = stream_assistant_response(
        client, request.query, chunks, history, system_prompt
    )
    if not is_stream:
        yield sse_event("answer", {"delta": result})
        yield sse_event("final", {"answer": result, "citations": []})
        yield sse_event("done", {})
        return

    response = ""
    previous_order: list[int] = []
    used_chunks: list[CitationChunk] = []
    for event in result:
        delta = event.choices[0].delta.content
        if not delta:
            continue
        response += delta
        yield sse_event("answer", {"delta": delta})
        order = extract_citation_order(response)
        if order != previous_order:
            available_indices = {chunk.index for chunk in chunks}
            filtered_order = [idx for idx in order if idx in available_indices]
            if filtered_order:
                used_chunks = select_citations_in_order(chunks, filtered_order)
                display_map = {orig: idx + 1 for idx, orig in enumerate(filtered_order)}
                for chunk in used_chunks:
                    chunk.display_index = display_map.get(chunk.index)
                yield sse_event(
                    "citations", {"citations": serialize_citations(used_chunks)}
                )
            previous_order = order

    assistant_text, used_chunks = apply_citation_display(response, chunks)
    yield sse_event(
        "final", {"answer": assistant_text, "citations": serialize_citations(used_chunks)}
    )

    yield sse_event("done", {})


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        iter_stream_events(request),
        media_type="text/event-stream",
    )


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}
