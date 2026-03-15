"""
tools/healthcare_db_tool.py
---------------------------
Tool 1 — Healthcare Knowledge Database Tool

Wraps the local dataset search as a LangChain Tool.
The agent calls this tool when it needs disease/symptom information from the
HuggingFace healthcare dataset.
"""

from langchain.tools import Tool
from dataset_loader import search_symptoms


def _run_db_search(query: str) -> str:
    """Entry point called by the LangChain agent."""
    result = search_symptoms(query, top_k=2)
    # Always append a safety reminder in the tool output
    disclaimer = (
        "\n\n⚠️  The above is general healthcare information only. "
        "It does NOT constitute a medical diagnosis. "
        "Please consult a qualified healthcare professional for personal advice."
    )
    return result + disclaimer


# Create the LangChain Tool object
healthcare_db_tool = Tool(
    name="HealthcareKnowledgeDB",
    func=_run_db_search,
    description=(
        "Search the local healthcare knowledge database for disease and symptom "
        "information. Input should be a string of symptom keywords "
        "(e.g. 'fever headache fatigue'). Returns relevant disease descriptions "
        "and general information. Use this tool FIRST when a user mentions symptoms."
    ),
)
