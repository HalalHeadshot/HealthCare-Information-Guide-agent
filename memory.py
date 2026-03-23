"""
memory.py
---------
Provides short-term conversation memory for the Healthcare AI Agent.

Uses LangChain ConversationBufferMemory to retain:
  - symptoms mentioned earlier in the conversation
  - previous questions and context
  - full chat history for multi-turn reasoning

The memory is passed to the AgentExecutor so the agent can reference
earlier turns without the user needing to repeat themselves.
"""

from langchain.memory import ConversationBufferMemory


def create_memory() -> ConversationBufferMemory:
    """
    Instantiate and return a fresh ConversationBufferMemory.

    memory_key  : matches the {chat_history} variable in the agent prompt
    return_messages : stores messages as HumanMessage / AIMessage objects
                      rather than plain strings (required for chat models)
    """
    return ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,       # emit HumanMessage/AIMessage objects for ChatPromptTemplate
        input_key="input",          # maps to the {input} prompt variable
        output_key="output",        # maps to the agent's final answer
    )
