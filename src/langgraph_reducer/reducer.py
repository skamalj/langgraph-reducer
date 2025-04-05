from langchain_core.messages import AIMessage, ToolMessage, HumanMessage, RemoveMessage, SystemMessage
from typing_extensions import Annotated, TypedDict
from typing import Callable, Optional, List, Dict, Any



class Reducer:
    def __init__(self, min_messages=0, max_messages=None):
        super().__init__()
        self.min_messages = min_messages
        self.max_messages = max_messages

    def reduce_messages(self, messages=[], message=None):
        
        messages = messages + message
        if self.max_messages is None or len(messages) <= self.max_messages:
            return messages
        # Identify AIMessage and HumanMessage indices to prune
        to_delete = set()
        # Calculate excess messages to remove
        excess_count = len(messages) - self.min_messages

        for i, msg in enumerate(messages[1:excess_count], start=1):
            if isinstance(msg, (AIMessage, HumanMessage)):
                to_delete.add(i)


            # If AIMessage, find and mark associated ToolMessages
            if isinstance(messages[i], AIMessage) and hasattr(messages[i], 'tool_calls'):
                for tool_call in messages[i].tool_calls:
                    tool_call_id = tool_call.get("id")
                    for j in range(i + 1, len(messages)):
                        if isinstance(messages[j], ToolMessage) and messages[j].tool_call_id == tool_call_id:
                            to_delete.add(j)
                            
        # Delete messages in reverse order to avoid index shifting
        for idx in sorted(to_delete, reverse=True):
            del messages[idx]
        print(f"Reduced messages from {len(messages) + excess_count} to {len(messages)}: {messages}")
        return messages

class PrunableStateFactory:
    @staticmethod
    def create_prunable_state(min_messages: int, max_messages: int):
        reducer = Reducer(min_messages=min_messages, max_messages=max_messages)
        
        class PrunableMessageState(TypedDict):
            messages: Annotated[list, reducer.reduce_messages]

        return PrunableMessageState

class MessagePrunerNode:
    def __init__(
        self,
        min_messages: int = 0,
        max_messages: Optional[int] = None,
        model_func: Optional[Callable[[List[Any]], Any]] = None,
    ):
        self.min_messages = min_messages
        self.max_messages = max_messages
        self.model_func = model_func

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if self.max_messages is None or len(messages) <= self.max_messages:
            return state  # Nothing to prune

        pruned_messages = []
        remove_list = []
        excess_count = len(messages) - self.min_messages

        for i, msg in enumerate(messages[1:excess_count], start=1):
            if isinstance(msg, (AIMessage, HumanMessage)) and hasattr(msg, "id"):
                remove_list.append(RemoveMessage(id=msg.id))
                pruned_messages.append(msg)

            if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls"):
                for tool_call in msg.tool_calls:
                    tool_call_id = tool_call.get("id")
                    for j in range(i + 1, len(messages)):
                        potential_tool_msg = messages[j]
                        if (
                            isinstance(potential_tool_msg, ToolMessage)
                            and potential_tool_msg.tool_call_id == tool_call_id
                            and hasattr(potential_tool_msg, "id")
                        ):
                            remove_list.append(RemoveMessage(id=potential_tool_msg.id))
                            pruned_messages.append(potential_tool_msg)

        result = {}

        if pruned_messages and self.model_func:
            summarizer_response = self.summarize(state, pruned_messages)
            result["summary"] = summarizer_response["summary"]

        result["messages"] = remove_list
        return result

    def summarize(self, state:  Dict[str, Any], messages_to_summarize: List[Any]) -> Dict[str, Any]:
        """
        Summarize the conversation in English.
        If an existing summary exists, extend it with new messages.
        Delete older messages to reduce context usage,
        keeping only the last two messages.
        """
        summary = state.get("summary", "")

        if summary:
            summary_message = (
                f"This is the conversation summary so far:\n{summary}\n\n"
                "Please extend the summary by incorporating any new messages."
            )
        else:
            summary_message = "Please create a concise summary of the conversation so far."

        messages = messages_to_summarize + [HumanMessage(content=summary_message)]
        response = self.model_func(messages)

        return {"summary": response.content}
