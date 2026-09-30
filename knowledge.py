"""
knowledge.py — the RAG (Retrieval-Augmented Generation) part of the Khata Assistant.

RAG pipeline:
  1. LOAD     read shop_rules.md
  2. SPLIT    cut it into small chunks        (RecursiveCharacterTextSplitter)
  3. EMBED    turn each chunk into numbers    (FastEmbed, runs locally, free)
  4. STORE    keep the vectors in memory      (InMemoryVectorStore)
  5. RETRIEVE find the chunks closest to a question (retriever.invoke)
The agent calls search_shop_rules() as a tool, then uses what it found
to write its answer (the "augmented generation" part).

Backup: if the embedding model cannot be downloaded (no internet in class),
we fall back to BM25 keyword search, so the demo still works.
"""

import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

RULES_FILE = Path(__file__).parent / "shop_rules.md"


def build_retriever(k: int = 3):
    # 1. LOAD
    text = RULES_FILE.read_text(encoding="utf-8")
    docs = [Document(page_content=text, metadata={"source": RULES_FILE.name})]

    # 2. SPLIT — separators try headings first, then paragraphs, then lines
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=40, separators=["\n## ", "\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(docs)

    # 3 + 4. EMBED and STORE
    try:
        from langchain_community.embeddings import FastEmbedEmbeddings
        from langchain_core.vectorstores import InMemoryVectorStore

        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        store = InMemoryVectorStore.from_documents(chunks, embeddings)
        print(f"[RAG] {len(chunks)} chunks embedded with FastEmbed (vector search).")
        return store.as_retriever(search_kwargs={"k": k})
    except Exception as err:  # no internet / fastembed not installed
        from langchain_community.retrievers import BM25Retriever

        print(f"[RAG] Vector search unavailable ({type(err).__name__}); using BM25 keyword search.")
        return BM25Retriever.from_documents(chunks, k=k, preprocess_func=_words)


def _words(text: str) -> list[str]:
    """Lowercase, drop punctuation, and trim simple endings so that
    'Cigarettes,' matches 'cigarette' and 'collected' matches 'collect'."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [re.sub(r"(ing|ed|es|s)$", "", w) if len(w) > 4 else w for w in words]


_retriever = None


def load_knowledge():
    """Build the retriever once at startup, so the first question is fast."""
    global _retriever
    if _retriever is None:
        _retriever = build_retriever()


@tool
def search_shop_rules(question: str) -> str:
    """Search the shop's rules document. Use for questions about credit limits,
    payment collection days, discounts, items that are cash-only, opening
    hours, or returns. Also use before giving credit, to check the limit."""
    load_knowledge()
    # 5. RETRIEVE
    results = _retriever.invoke(question)
    if not results:
        return "No matching rule found."
    return "\n---\n".join(d.page_content.strip() for d in results)
