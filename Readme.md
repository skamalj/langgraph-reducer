# langgraph-reducer

This package provides:
- 🧠 `PrunableStateFactory`: Auto-prunes message history without extra effort, works as a drop-in replacement for `MessageState`.
- 🧹 `MessagePrunerNode`: A LangGraph node that summarizes or deletes messages when they exceed a count.
- ⚠️ Important difference:  
  - `PrunableState` runs **on every node execution** — better for lightweight, fast pruning.
  - `MessagePrunerNode` runs **only where you add it** — better for summarization or heavier logic (e.g., calling LLMs).

---

## 🚀 Install

```bash
pip install langgraph-reducer
```

---

## 🔁 PrunableState — Auto Cleanup

Use this when you just want to keep the latest N messages and forget the rest. No node logic needed — pruning is automatic.

```python
from langraph_reducer import PrunableStateFactory

# Parameters:
# - min_messages: number of most recent messages to retain
# - max_messages: pruning is triggered once this threshold is exceeded

PrunableMessagesState = PrunableStateFactory.create_prunable_state(
    min_messages=10,
    max_messages=15
)
```

Use the `PrunableMessagesState` as the state type in your LangGraph:

```python
from langgraph.graph import StateGraph

graph = StateGraph(PrunableMessagesState)
```

That’s it — pruning will be handled automatically during graph execution.

---

## 🧠 MessagePrunerNode — Summary or Delete

Use this when you want to **summarize or remove old messages** but need control over where in your graph this happens (e.g., before `END`).

```python
from langchain_openai import ChatOpenAI
from langgraph_reducer import MessagePrunerNode

llm = ChatOpenAI(model="gpt-4o")

# model_func: Optional. If provided, node summarizes older messages.
#             If not provided, node simply deletes older messages.

def model_func(messages):
    return llm.invoke(messages)

pruner_node = MessagePrunerNode(
    min_messages=4,       # Retain at least 4 most recent messages
    max_messages=6,       # Trigger pruning if messages exceed 6
    model_func=model_func # Optional. Use LLM to summarize old messages.
)
```

Then wire it into your graph like a normal LangGraph node:

```python
workflow.add_node("summarize_conversation", pruner_node)
workflow.add_edge("summarize_conversation", END)
```

Use a conditional edge to trigger the pruner only when needed.

---

## 🧩 Extending Prunable State

Want to add custom fields to the state? Just subclass the factory-created state:

```python
from typing import Annotated
from langgraph.graph.message import add_messages

# Base state with auto-pruning
BaseState = PrunableStateFactory.create_prunable_state(10, 15)

# Extend it with your own fields
class MyState(BaseState):
    summary: str = ""
    profile_id: str = ""
```

Use `MyState` in your graph instead of the base one.


⚠️ **Important:**  
Do **not** override or redefine the `messages` field when subclassing a prunable state.  
The `messages` field is **automatically managed** for pruning and must retain its behavior.

If you override it, pruning logic will break silently.

---

## 📓 Examples

- `notebooks/basic_pruner_node.ipynb`: shows how to integrate a summarizing node.
- `scripts/prunable_state_lambda.py`: shows full pipeline with `PrunableMessagesState` in AWS Lambda.

---

