# Healthcare Information Guide Agent

An LLM-powered ReAct agent that provides general healthcare guidance based on user-described symptoms. Built with LangChain, the HuggingFace `harmesh95/healthcare-disease-knowledge` dataset, and the Groq API.

> ⚠️ **Disclaimer**: This system provides general health information only. It is **not** a diagnostic tool and does **not** replace professional medical advice. Always consult a qualified healthcare professional.

---

## Project Structure

```
Healthcare-AI/
├── main.py                  # CLI entry point
├── agent.py                 # LangChain ReAct agent
├── memory.py                # ConversationBufferMemory
├── dataset_loader.py        # HuggingFace dataset loader + search
├── reasoning_logger.py      # JSON reasoning trace logger
├── tools/
│   ├── __init__.py
│   ├── healthcare_db_tool.py  # Tool 1 — local dataset search
│   └── web_search_tool.py     # Tool 2 — web medical guideline search (ddgs scraper)
├── logs/                    # Auto-created; session reasoning traces saved here
├── requirements.txt
├── .env.example
└── README.md
```

---

## Prerequisites

- Python 3.10 or later
- A [Groq API key](https://console.groq.com/keys)

---

## Installation

```bash
# 1. Clone / navigate to the project
cd /Users/sofian/Documents/Healthcare-AI

# 2. (Recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS / Linux

# 3. Install all dependencies
pip install -r requirements.txt
```

---

## Configuration

```bash
# Copy the template
cp .env.example .env

# Open .env and fill in your key
# GROQ_API_KEY=gsk_...
# GROQ_MODEL=llama3-8b-8192   (optional, default is llama3-8b-8192)
```

---

## Running the Agent

```bash
python main.py
```

The first run will download the HuggingFace dataset (~a few MB). Subsequent runs use the cache.

---

## Example Session

```
You: I have a fever and a headache

[Agent — Turn 1] Thinking …

Thought: Identify symptoms: fever, headache. Search local DB first.
Tool: HealthcareKnowledgeDB
Tool Input: fever headache
Observation: [disease records from dataset]

Thought: Now get official treatment guidelines.
Tool: WebMedicalSearch
Tool Input: WHO fever headache treatment guidelines
Observation: [WHO/CDC snippets]

Final Answer: Based on the healthcare database and WHO guidelines, fever 
combined with headache may be associated with influenza or other infections. 
Recommended care: rest, hydration, and over-the-counter fever reducers. 
See a doctor if fever exceeds 39°C, persists more than 3 days, or is 
accompanied by stiff neck or rash. ⚠ DISCLAIMER: …
```

---

## Reasoning Logs

Every session saves a structured JSON log to `logs/session_<timestamp>.json`:

```json
[
  {
    "session_id": "20240315_190500",
    "turn": 1,
    "user_input": "I have a fever and a headache",
    "intermediate_steps": [
      {
        "thought": "Identify symptoms …",
        "tool": "HealthcareKnowledgeDB",
        "tool_input": "fever headache",
        "observation": "…"
      }
    ],
    "final_answer": "…"
  }
]
```

---

## Safety & Limitations

| Rule | Implementation |
|------|---------------|
| No medical diagnosis | Enforced in system prompt |
| Always recommend doctor for serious symptoms | Hard-coded in prompt |
| General guidance only | Safety disclaimer in every final answer |
| No personally identifiable data collected | Local memory only; clears on restart |
