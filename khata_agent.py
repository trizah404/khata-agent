"""
Khata Assistant — a LangGraph credit-ledger agent for small shops.

Shopkeepers type in plain language (English or Nepali-English mix):
    "Ram took Rs 500 of rice on credit"
    "Sita paid 200"
    "How much does Ram owe?"
    "What is the credit limit?"          <- answered with RAG from shop_rules.md
The AI decides which tool to use: ledger tools update a SQLite database,
and the RAG tool searches the shop's rules document. The AI then replies
using the tool results. Conversation memory is kept per session.

LangChain / LangGraph concepts used:
  - RAG          : shop_rules.md -> split -> embed -> vector store -> retrieve (knowledge.py)
  - Tools        : 4 ledger tools + 1 RAG search tool
  - State        : the list of messages shared between steps
  - Nodes        : "assistant" (the AI thinks) and "tools" (Python runs)
  - Conditional edge : after the AI thinks, go to "tools" OR finish
  - Loop         : tools -> assistant, so the AI can read the tool result
  - Checkpointer : MemorySaver remembers the conversation (thread_id)
"""

import os
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from knowledge import load_knowledge, search_shop_rules

load_dotenv()
DB_PATH = "khata.db"


# ---------------------------------------------------------------- database
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS entries (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               customer TEXT NOT NULL,
               amount REAL NOT NULL,      -- positive = credit taken, negative = payment
               note TEXT,
               created_at TEXT NOT NULL)"""
    )
    return conn


def _balance(conn, customer: str) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) FROM entries WHERE customer = ?",
        (customer,),
    ).fetchone()
    return row[0]


# ---------------------------------------------------------------- tools
# The docstrings matter: the AI reads them to decide which tool to call.
@tool
def add_credit(customer: str, amount: float, note: str = "") -> str:
    """Record that a customer took goods on credit (udhaaro). Use when someone
    takes items without paying. amount is in rupees and must be positive."""
    if amount <= 0:
        return "Error: amount must be positive."
    name = customer.strip().title()
    with db() as conn:
        conn.execute(
            "INSERT INTO entries (customer, amount, note, created_at) VALUES (?,?,?,?)",
            (name, amount, note, datetime.now().isoformat(timespec="minutes")),
        )
        return f"Saved: {name} took Rs {amount:.0f} on credit. New balance: Rs {_balance(conn, name):.0f}."


@tool
def record_payment(customer: str, amount: float) -> str:
    """Record that a customer paid back money they owed. amount is in rupees
    and must be positive."""
    if amount <= 0:
        return "Error: amount must be positive."
    name = customer.strip().title()
    with db() as conn:
        owed = _balance(conn, name)
        if owed <= 0:
            return f"{name} does not owe anything, so no payment was recorded."
        if amount > owed:
            return f"{name} only owes Rs {owed:.0f}, but the payment was Rs {amount:.0f}. Please check the amount."
        conn.execute(
            "INSERT INTO entries (customer, amount, note, created_at) VALUES (?,?,?,?)",
            (name, -amount, "payment", datetime.now().isoformat(timespec="minutes")),
        )
        return f"Saved: {name} paid Rs {amount:.0f}. Remaining balance: Rs {_balance(conn, name):.0f}."


@tool
def get_balance(customer: str) -> str:
    """Look up how much one customer currently owes, with their recent entries."""
    name = customer.strip().title()
    with db() as conn:
        rows = conn.execute(
            "SELECT amount, note, created_at FROM entries WHERE customer = ? "
            "ORDER BY id DESC LIMIT 5",
            (name,),
        ).fetchall()
        if not rows:
            return f"No records found for {name}."
        history = "; ".join(f"{d}: {'+' if a > 0 else ''}{a:.0f} {n or ''}".strip() for a, n, d in rows)
        return f"{name} owes Rs {_balance(conn, name):.0f}. Recent: {history}"


@tool
def list_all_dues() -> str:
    """List every customer who still owes money, highest first, with the total."""
    with db() as conn:
        rows = conn.execute(
            "SELECT customer, SUM(amount) AS due FROM entries GROUP BY customer "
            "HAVING due > 0 ORDER BY due DESC"
        ).fetchall()
    if not rows:
        return "Nobody owes anything right now."
    lines = [f"{c}: Rs {d:.0f}" for c, d in rows]
    return "\n".join(lines) + f"\nTotal outstanding: Rs {sum(d for _, d in rows):.0f}"


TOOLS = [add_credit, record_payment, get_balance, list_all_dues, search_shop_rules]


# ---------------------------------------------------------------- model
def get_llm():
    """Use Groq (key starts with gsk_) or xAI Grok (key starts with xai-)."""
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("MODEL", "llama-3.3-70b-versatile"), temperature=0)
    if os.getenv("XAI_API_KEY"):
        from langchain_xai import ChatXAI
        return ChatXAI(model=os.getenv("MODEL", "grok-3-mini"), temperature=0)
    raise SystemExit("No API key found. Put GROQ_API_KEY or XAI_API_KEY in a .env file.")


SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a friendly khata (credit ledger) assistant for a small shop in Nepal. "
        "Always use the tools to save or look up records; never guess balances. "
        "For any question about shop policy (credit limits, collection days, discounts, "
        "cash-only items, opening hours, returns), call search_shop_rules and answer "
        "only from what it returns. After adding credit, check the credit limit with "
        "search_shop_rules and warn the shopkeeper if the new balance is over the limit. "
        "Amounts are in Nepali rupees. If the customer name or amount is missing, "
        "ask for it instead of calling a tool. Keep replies short and clear."
    )
)


# ---------------------------------------------------------------- graph
def build_graph(llm):
    llm_with_tools = llm.bind_tools(TOOLS)

    def assistant(state: MessagesState):
        # Node 1: the AI reads the conversation and either replies or asks for a tool.
        response = llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
        return {"messages": [response]}

    def route(state: MessagesState):
        # Conditional edge: did the AI ask to use a tool?
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(MessagesState)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(TOOLS))          # Node 2: runs the Python tools
    graph.add_edge(START, "assistant")
    graph.add_conditional_edges("assistant", route, ["tools", END])
    graph.add_edge("tools", "assistant")              # loop back so the AI sees the result
    return graph.compile(checkpointer=MemorySaver())  # remembers the chat per thread


# ---------------------------------------------------------------- chat loop
def main():
    app = build_graph(get_llm())
    load_knowledge()  # build the RAG index once, before chatting
    config = {"configurable": {"thread_id": "shop-1"}}
    print("Khata Assistant ready. Type 'quit' to exit.\n")
    while True:
        text = input("You: ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            break
        if not text:
            continue
        result = app.invoke({"messages": [HumanMessage(content=text)]}, config)
        print(f"Khata: {result['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
