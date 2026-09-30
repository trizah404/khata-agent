# Khata Assistant — a LangGraph Agent with RAG


## Architecture

```mermaid
graph LR
    START --> assistant
    assistant -- "AI wants a tool" --> tools
    tools --> assistant
    assistant -- "AI has the answer" --> END
```

