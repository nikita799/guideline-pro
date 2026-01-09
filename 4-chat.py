import os
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv
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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
COLLECTION_NAME = "Guideline"


@dataclass
class CitationChunk:
    index: int
    search_text: str
    source: str | None
    year: str | None
    breadcrumbs: str | None

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


def retrieve_chunks(query: str, k: int = 5, alpha: float = 0.8) -> List[CitationChunk]:
    with weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    ) as client:
        col = client.collections.use(COLLECTION_NAME)
        props = ["search_text", "breadcrumbs", "chunk_id", "source", "year"]
        resp = col.query.hybrid(query=query, alpha=alpha, limit=k, return_properties=props)

    chunks: List[CitationChunk] = []
    for idx, obj in enumerate(resp.objects, start=1):
        props = obj.properties or {}
        chunks.append(
            CitationChunk(
                index=idx,
                search_text=props.get("search_text", ""),
                source=props.get("source"),
                year=props.get("year"),
                breadcrumbs=props.get("breadcrumbs"),
            )
        )
    return chunks


def build_context(chunks: List[CitationChunk]) -> str:
    if not chunks:
        return ""
    lines = []
    for chunk in chunks:
        lines.append(f"[{chunk.index}] {chunk.search_text}")
    return "\n\n".join(lines)


def format_citation_links(chunks: List[CitationChunk]) -> str:
    if not chunks:
        return ""
    links = " ".join([f"[{chunk.index}](#citation-{chunk.index})" for chunk in chunks])
    return links


def get_assistant_response(client: OpenAI, query: str, chunks: List[CitationChunk], history: List[dict]) -> str:
    context = build_context(chunks)
    if not context:
        return (
            "I couldn't find relevant guideline passages for that question. "
            "Please try a more specific clinical query."
        )

    system_prompt = (
        "You are a clinical guideline assistant. Answer the clinician's question "
        "using only the provided guideline excerpts. Do not add knowledge beyond "
        "the excerpts. If the excerpts do not contain the answer, say you do not "
        "have enough information. Provide concise, clinically relevant answers. "
        "Cite sources minimally in square brackets like [1]."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "system",
            "content": f"Guideline excerpts:\n{context}",
        },
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content


def render_citations(chunks: List[CitationChunk]) -> None:
    if not chunks:
        return
    st.markdown("### Citations")
    for chunk in chunks:
        st.markdown(
            f"<a name='citation-{chunk.index}'></a>",
            unsafe_allow_html=True,
        )
        with st.expander(f"[{chunk.index}] {chunk.label}"):
            st.write(chunk.search_text)


def main() -> None:
    st.set_page_config(page_title="Clinical Guideline Chat", page_icon="🩺")
    st.title("🩺 Clinical Guideline Chat")
    st.caption("Ask clinical questions and receive guideline-based answers with citations.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    query = st.chat_input("Ask a clinical question")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        client = OpenAI()
        chunks = retrieve_chunks(query)
        history = [
            msg
            for msg in st.session_state.messages
            if msg["role"] in {"user", "assistant"}
        ]
        response = get_assistant_response(client, query, chunks, history)
        citation_links = format_citation_links(chunks)
        assistant_content = response
        if citation_links:
            assistant_content = f"{response}\n\n{citation_links}"

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_content}
        )
        with st.chat_message("assistant"):
            st.markdown(assistant_content, unsafe_allow_html=True)
        render_citations(chunks)


if __name__ == "__main__":
    main()
