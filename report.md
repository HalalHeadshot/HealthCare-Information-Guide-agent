# Healthcare Information Guide Agent
## Academic Project Report

---

## 1. Abstract

This report describes the design and implementation of a **Healthcare Information Guide Agent**, an AI-powered conversational system that assists users in understanding basic healthcare guidance based on self-reported symptoms. The agent is built on a **LangChain ReAct (Reasoning + Acting) architecture** powered by a large language model (LLM) and integrates two tools: a local disease knowledge database derived from the HuggingFace `harmesh95/healthcare-disease-knowledge` dataset, and a web-based medical guideline retrieval tool querying trusted sources such as the WHO, CDC, Mayo Clinic, and MedlinePlus. The agent maintains short-term conversation memory, exposes a transparent reasoning trace for auditability, and enforces strict safety rules that prohibit medical diagnosis. This work demonstrates how modern LLM agent frameworks can be responsibly applied to the healthcare information domain.

---

## 2. Introduction

The rapid advancement of large language models has created new opportunities to make healthcare information more accessible to the general public. Traditional rule-based health chatbots are limited by rigid decision trees and cannot adapt to the nuanced, open-ended nature of patient questions. LLM-powered agents, by contrast, can plan reasoning steps, dynamically select tools, and synthesise information from multiple sources.

This project applies these capabilities to a constrained, safety-conscious use case: helping users understand what their symptoms might indicate and when they should seek professional medical attention. The system explicitly stops short of providing diagnoses, deferring all clinical judgement to qualified healthcare professionals.

---

## 3. Problem Statement

**Challenge**: Most people lack access to reliable, personalised health information. Existing resources are either too technical (clinical literature), too generic (FAQ pages), or too unreliable (general web searches).

**Proposed solution**: An LLM agent that:
1. Engages in natural conversation about symptoms.
2. Retrieves relevant disease information from a curated dataset.
3. Fetches current guidelines from authoritative medical sources.
4. Synthesises a safe, informative response with appropriate caveats.
5. Remembers context within a session so users do not need to repeat themselves.

**Scope**: The system provides *general informational guidance only*. It is out of scope for the system to diagnose, prescribe, or replace clinical consultation.

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────┐
│                      User (CLI)                     │
└──────────────────────────┬──────────────────────────┘
                           │  natural language query
                           ▼
┌─────────────────────────────────────────────────────┐
│                    LangChain Agent                  │
│  ┌───────────────────────────────────────────────┐  │
│  │  ReAct Prompt  (safety rules + ReAct format)  │  │
│  └───────────────────────────────────────────────┘  │
│  ┌────────────────────┐  ┌────────────────────────┐ │
│  │  ChatGroq (LLM)    │  │  ConversationBuffer    │ │
│  │  llama3-8b-8192    │  │  Memory                │ │
│  └────────────────────┘  └────────────────────────┘ │
└──────────────┬────────────────────────┬─────────────┘
               │                        │
     ┌─────────▼──────────┐  ┌─────────▼──────────────┐
     │ Tool 1             │  │ Tool 2                  │
     │ HealthcareKnowledgeDB │ WebMedicalSearch       │
     │                    │  │                         │
     │ HuggingFace        │  │ DuckDuckGo API →        │
     │ healthcare dataset │  │ WHO / CDC / Mayo /      │
     │ (local, pandas)    │  │ MedlinePlus (web)       │
     └─────────┬──────────┘  └─────────┬───────────────┘
               │                        │
               └───────────┬────────────┘
                           ▼
              ┌───────────────────────┐
              │   Reasoning Logger    │
              │   logs/session_*.json │
              └───────────────────────┘
                           │
                           ▼
                   Final Answer + Disclaimer
                        → User
```

---

## 5. Agent Design

### 5.1 ReAct Reasoning Pattern

The system implements the **ReAct** (Reason + Act) pattern introduced by Yao et al. (2022). This interleaves reasoning traces with tool calls, enabling the agent to act on intermediate observations before producing a final answer.

**Step sequence**:

```
User Goal
   ↓
Thought  : Analyse the symptom description and plan next step.
   ↓
Action   : Select a tool (HealthcareKnowledgeDB or WebMedicalSearch).
   ↓
Observation : Receive and interpret tool output.
   ↓
Next Thought : Decide whether more information is needed or produce the final answer.
   ↓
Final Answer : Safe, informative response with disclaimer.
```

### 5.2 LangChain Implementation

The agent is constructed using `create_react_agent` from `langchain.agents`. The `AgentExecutor` wrapper manages the tool-calling loop, memory injection, and error recovery.

| Component | Implementation |
|-----------|---------------|
| LLM | `ChatGroq` (`llama3-8b-8192`, temp=0.3) |
| Agent type | `create_react_agent` (ReAct) |
| Tool dispatch | Dynamic — LLM decides which tool and when |
| Memory | `ConversationBufferMemory` |
| Error handling | `handle_parsing_errors=True` |
| Max iterations | 8 (prevents infinite loops) |

---

## 6. Tools Used

### Tool 1 — HealthcareKnowledgeDB

| Property | Value |
|----------|-------|
| **Name** | `HealthcareKnowledgeDB` |
| **Source** | HuggingFace `harmesh95/healthcare-disease-knowledge` |
| **Method** | Keyword scoring over all text columns (pandas) |
| **Output** | Top-3 matching disease records (name, symptoms, description) |
| **API key** | None required |

The dataset is downloaded once and cached locally. A keyword scoring function ranks rows by how many query terms appear in the combined text of all columns. The top-3 ranked rows are returned as formatted text.

### Tool 2 — WebMedicalSearch

| Property | Value |
|----------|-------|
| **Name** | `WebMedicalSearch` |
| **Source** | DuckDuckGo Instant Answer API |
| **Filter** | Biased toward who.int, cdc.gov, mayoclinic.org, medlineplus.gov |
| **Output** | Top-3 result titles, snippets, and URLs |
| **API key** | None required |

The DuckDuckGo API is free and requires no authentication. The query is augmented with site-filter keywords before submission to bias results toward authoritative medical sources.

---

## 7. Memory Design

The agent uses **`ConversationBufferMemory`** from LangChain, which stores the complete history of human and AI messages for the duration of a session.

**Benefits**:
- Allows the user to mention additional symptoms in follow-up turns without repeating earlier context.
- Enables the agent to reference previously retrieved information.

**Limitations**:
- Memory is session-scoped and deleted when the CLI exits (by design, for privacy).
- Very long sessions may exceed the LLM's context window; mitigated by the 8-iteration cap.

**Memory variables**:
```
memory_key   = "chat_history"
return_messages = True
input_key    = "input"
output_key   = "output"
```

---

## 8. Planning Strategy (ReAct)

The agent's planning is governed by the **ReAct system prompt**. Key planning instructions embedded in the prompt:

1. **Tool selection order**: Use `HealthcareKnowledgeDB` first (local, fast) then `WebMedicalSearch` for official guidelines.
2. **Stopping condition**: When the agent's thought contains "I now have enough information" it produces the final answer.
3. **Safety override**: Safety rules are listed at the top of the prompt and are evaluated before any other reasoning.

This structured prompt guides the LLM to behave consistently without hard-coded decision trees.

---

## 9. Implementation Details

### 9.1 Dataset Loading (`dataset_loader.py`)

- Uses `datasets.load_dataset()` with `trust_remote_code=True`.
- Combines all dataset splits into a single `pandas.DataFrame`.
- Column names are normalised (lowercase, underscores).
- `search_symptoms(query)` splits the query into keywords (≥3 chars) and scores rows.

### 9.2 Tool Integration (`tools/`)

- Both tools are `langchain.tools.Tool` instances with `name`, `func`, and `description`.
- Descriptions are carefully written so the LLM knows *when* to use each tool.
- Tool outputs always include a safety disclaimer.

### 9.3 Agent Construction (`agent.py`)

- `PromptTemplate` defines the full ReAct format including safety rules.
- `AgentExecutor` with `return_intermediate_steps=True` exposes all reasoning steps.
- `verbose=True` prints live reasoning to the terminal.

### 9.4 Reasoning Logger (`reasoning_logger.py`)

- `ReasoningLogger` creates a unique session file (`logs/session_<timestamp>.json`).
- `log_turn()` extracts `(AgentAction, observation)` pairs and structures them as JSON.
- `print_trace()` renders a human-readable trace to stdout after each turn.

### 9.5 CLI Interface (`main.py`)

- Displays a startup banner with the safety disclaimer.
- Input loop with graceful EOF / KeyboardInterrupt handling.
- Error messages are user-friendly and also persisted to the log.

---

## 10. Safety Considerations

| Risk | Mitigation |
|------|-----------|
| Providing a medical diagnosis | Hard-coded prohibition in system prompt |
| User ignores safety advice | Disclaimer printed in every final answer |
| Hallucinated medical facts | Web tool retrieves from authoritative sources |
| Infinite reasoning loops | `max_iterations=8` in AgentExecutor |
| LLM output format errors | `handle_parsing_errors=True` |
| Session data privacy | Memory is in-process only; no persistent user data |

**Disclaimer text** (included in every agent response):
> "⚠ DISCLAIMER: This information is for educational purposes only and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional."

---

## 11. Agent Interaction Examples

### Example 1 — Fever and Headache

```
User: I have fever and headache

Thought: The user reports fever and headache. I should search the
         healthcare database first.

Action: HealthcareKnowledgeDB
Action Input: fever headache

Observation:
  --- Result 1 ---
  Disease: Influenza (Flu)
  Symptoms: Fever, headache, body aches, fatigue, cough
  Description: A contagious respiratory illness caused by influenza viruses.

Thought: I have dataset results. Now I will get WHO/CDC guidelines.

Action: WebMedicalSearch
Action Input: WHO flu fever headache treatment guidelines

Observation:
  [1] Influenza (Flu) — CDC
      Snippet: Rest and fluids are recommended. Antiviral drugs may
      shorten illness if taken early. See a doctor if symptoms worsen.
      URL: https://www.cdc.gov/flu/...

Final Answer:
Fever combined with headache is commonly associated with influenza (flu)
or other viral infections.

General care advice:
• Rest and stay hydrated (8+ glasses of water/day).
• Over-the-counter medications (e.g., paracetamol/ibuprofen) can reduce fever.
• Avoid public spaces to prevent spreading infection.

See a doctor immediately if:
• Fever exceeds 39°C (102°F) or persists more than 3 days.
• You develop a stiff neck, rash, confusion, or difficulty breathing.
• Symptoms worsen rather than improve after 48 hours.

⚠ DISCLAIMER: This information is for educational purposes only …
```

### Example 2 — Follow-up Question (Memory Demo)

```
User: Should I take antibiotics?

Thought: The user is asking about antibiotics for the fever and headache
         mentioned earlier (from memory). Flu is viral — antibiotics do
         not help. I should advise correctly.

Final Answer:
Antibiotics are NOT effective against viral infections like influenza.
Taking antibiotics unnecessarily can cause antibiotic resistance.
Your symptoms suggest a viral illness — antibiotics should not be taken
unless a doctor specifically prescribes them after examination.

Please consult a healthcare professional who can determine whether a
bacterial infection is present and whether antibiotics are appropriate.

⚠ DISCLAIMER: This information is for educational purposes only …
```

---

## 12. Comparative Analysis

### Classical Agent vs. Modern LLM Agent

| Feature | Rule-Based / Classical Agent | LLM Agent (this project) |
|---------|------------------------------|--------------------------|
| **Decision Making** | Follows hard-coded if/else rules | Reasons dynamically using natural language |
| **Planning** | Predefined flowchart | Multi-step ReAct planning at inference time |
| **Memory** | Typically stateless or simple session variables | ConversationBufferMemory — full dialogue history |
| **Adaptability** | Cannot handle inputs outside the rule set | Handles unexpected phrasings and edge cases |
| **Reasoning** | None — deterministic path | Explicit Thought → Action → Observation chain |
| **Tool Integration** | Hard-coded API calls | Dynamic tool selection based on context |
| **Language Understanding** | Keyword matching / NLU pipelines | Full natural-language comprehension (LLM) |
| **Maintenance** | Requires manual rule updates | Improved via prompt engineering |
| **Transparency** | Rules are explicit | Reasoning trace logged for auditability |
| **Safety** | Rules must be manually enumerated | Safety constraints in system prompt |

**Key insight**: Classical agents are predictable and lightweight but brittle. LLM agents are flexible and can handle the open-ended nature of health conversations, but require careful prompt engineering to enforce safety boundaries.

---

## References

- Yao, S. et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models. *arXiv:2210.03629*.
- LangChain Documentation. https://python.langchain.com
- HuggingFace Dataset: `harmesh95/healthcare-disease-knowledge`. https://huggingface.co/datasets/harmesh95/healthcare-disease-knowledge
- World Health Organization. https://www.who.int
- Centers for Disease Control and Prevention. https://www.cdc.gov
- Mayo Clinic. https://www.mayoclinic.org
- MedlinePlus. https://medlineplus.gov
- Groq API. https://console.groq.com/docs
