"""
agent.py
--------
Builds and returns the LangChain ReAct agent with:
  • Two tools  : HealthcareKnowledgeDB, WebMedicalSearch
  • Memory     : ConversationBufferMemory (short-term, session-scoped)
  • Safety     : System prompt enforces no-diagnosis rule + disclaimer
  • Tracing    : verbose=True exposes Thought → Action → Observation chain

ReAct reasoning pattern
-----------------------
User Goal → Thought → Action (Tool) → Observation
          → Next Thought → … → Final Answer
"""

import os
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from memory import create_memory
from tools import ALL_TOOLS

# ---------------------------------------------------------------------------
# System prompt / ReAct template (ChatPromptTemplate)
# ---------------------------------------------------------------------------
# Using ChatPromptTemplate so that:
#   • The system instructions are sent as a SYSTEM role message (LLaMA-friendly)
#   • {chat_history} is injected as real HumanMessage/AIMessage objects via
#     MessagesPlaceholder — not as a plain concatenated string
#   • {input} + {agent_scratchpad} are in the final HUMAN turn
#
# Required template variables:
#   {tools}            — tool names + descriptions (auto-filled by create_react_agent)
#   {tool_names}       — comma-separated tool names (auto-filled)
#   {chat_history}     — list of message objects from ConversationBufferMemory
#   {input}            — current user message
#   {agent_scratchpad} — intermediate reasoning steps (auto-filled)

SYSTEM_TEMPLATE = """You are a Healthcare Information Guide Agent.

IMPORTANT SAFETY RULES (never violate these):
1. You MUST NOT provide any medical diagnosis.
2. You MUST always recommend consulting a qualified healthcare professional for serious symptoms.
3. You provide ONLY general health information and guidance.
4. Always include this disclaimer in every final answer:
   "⚠ DISCLAIMER: This information is for educational purposes only and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional."

YOUR REASONING PROCESS (ReAct pattern):
- Thought: Analyse the user's message and plan your next step.
- Action: Choose a tool to call.
- Action Input: The exact input to pass to the tool.
- Observation: Read and interpret the tool's output.
- Repeat Thought / Action / Observation as needed.
- Final Answer: Provide thorough, safe, and helpful guidance.

AVAILABLE TOOLS:
{tools}

FORMAT (strictly follow this):
Thought: <your reasoning here>
Action: <tool name — must be one of [{tool_names}]>
Action Input: <input string for the tool>
Observation: <tool result — filled automatically>
... (repeat as needed)
Thought: I now have enough information to answer.
Final Answer: <your safe, helpful response with disclaimer>

CRITICAL: Do NOT output "Action: None". If you are ready to answer, ONLY output "Final Answer: " followed by your answer!"""

PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE),
    MessagesPlaceholder(variable_name="chat_history"),   # injects real message objects
    HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}"),
])


def build_agent(verbose: bool = True) -> AgentExecutor:
    """
    Construct and return a ready-to-use AgentExecutor.

    Parameters
    ----------
    verbose : bool
        If True, LangChain prints every Thought/Action/Observation step
        to stdout in real time.

    Returns
    -------
    AgentExecutor
        The runnable agent with memory attached.
    """
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    llm = ChatGroq(
        model_name=model_name,
        temperature=0.3,          # low temperature for consistent, factual answers
        api_key=os.getenv("GROQ_API_KEY"),
    )

    # Core ReAct agent (stateless — memory lives in the executor)
    agent = create_react_agent(
        llm=llm,
        tools=ALL_TOOLS,
        prompt=PROMPT,
    )

    memory = create_memory()

    executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        memory=memory,
        verbose=verbose,            # exposes full reasoning trace in terminal
        handle_parsing_errors=True, # gracefully handle LLM output format errors
        max_iterations=5,           # prevent infinite loops
        max_execution_time=45,      # time limit on the execution
        return_intermediate_steps=True,  # needed by the reasoning logger
    )

    return executor
