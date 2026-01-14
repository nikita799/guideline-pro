import html
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
from langsmith import Client, traceable
from langsmith.wrappers import wrap_openai
from markdown_it import MarkdownIt
from openai import OpenAI
import streamlit as st
import weaviate
from weaviate.auth import Auth

load_dotenv()

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
SAMPLE_QUESTIONS = [
    "Millal alustada metformiinravi 2. tüüpi diabeedi korral?",
    "Millised on HbA1c eesmärgid 2. tüüpi diabeedi patsiendil?",
    "Kuidas valida farmakoteraapia 2. tüüpi diabeedis?",
]

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

MARKDOWN_RENDERER = MarkdownIt("commonmark", {"html": False, "linkify": True}).enable(
    "table"
)


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
    chat_top = "42vh" if not has_messages else "calc(100vh - 3.8rem)"
    sample_spacer_height = "28vh" if show_examples else "0"
    st.markdown(
        f"""
        <style>
        :root {{
          --chat-input-width: min(820px, 92vw);
        }}
        div[data-testid="stChatInput"] {{
          position: fixed;
          left: 50%;
          width: var(--chat-input-width);
          transform: translate(-50%, -50%);
          top: {chat_top};
          z-index: 1000;
          transition: top 0.6s ease;
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
    try:
        return MARKDOWN_RENDERER.render(text)
    except Exception:
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


@traceable_if_enabled("retrieve_chunks")
def retrieve_chunks(query: str, k: int = 5, alpha: float = 0.8) -> List[CitationChunk]:
    with weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    ) as client:
        col = client.collections.use(COLLECTION_NAME)
        props = ["text", "search_text", "breadcrumbs", "chunk_id", "source", "year"]
        resp = col.query.hybrid(query=query, alpha=alpha, limit=k, return_properties=props)

    chunks: List[CitationChunk] = []
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


def retrieve_chunks_fast(
    client: OpenAI, query: str, history: list[dict]
) -> List[CitationChunk]:
    rewritten_query: str | None = None
    chunks: List[CitationChunk] = []
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
    client: OpenAI,
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
    stream = client.chat.completions.create(
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
    show_examples = (not ui_has_messages) and (not st.session_state.examples_dismissed)
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
            followups = message.get("followups") or []
            if followups:
                render_question_chips(
                    followups,
                    key_prefix=f"followup-{render_token}-{group}",
                    label="Jätkuküsimused",
                )
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
        status_container = st.empty()
        with status_container.status("Loon vastust", expanded=True) as status:
            status.write("(1) Teen andmebaasi päringut")
            history = build_history(st.session_state.messages, query)
            chunks = retrieve_chunks_fast(client, query, history)
            status.write("(2) Loen vastuseid")
            system_prompt, _prompt_source = load_system_prompt(LANGSMITH_CLIENT)
            status.write("(3) Loon vastust")
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
                delta = event.choices[0].delta.content
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

        if assistant_text:
            followup_container = st.empty()
            followup_stream = stream_followup_questions(
                client, query, assistant_text, history
            )
            followup_raw = ""
            if followup_stream:
                followup_block = followup_container.container()
                followup_block.caption("Jätkuküsimused")
                preview_slot = followup_block.empty()
                preview_slot.markdown("_Koostan jätkuküsimusi..._")
                for event in followup_stream:
                    delta = event.choices[0].delta.content
                    if delta:
                        followup_raw += delta
                        preview_slot.markdown(format_followup_preview(followup_raw))
                followups = parse_followup_questions(followup_raw)
            else:
                followups = suggest_followup_questions(client, query, assistant_text, history)
            if followups:
                followup_container.empty()
                render_question_chips(
                    followups,
                    key_prefix=f"followup-{render_token}-{citation_group}",
                    label="Jätkuküsimused",
                    container=followup_container.container(),
                )
            else:
                followup_container.empty()

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
