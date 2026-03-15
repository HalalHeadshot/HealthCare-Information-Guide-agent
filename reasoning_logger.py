"""
reasoning_logger.py
-------------------
Captures and persists all agent reasoning steps (Thought, Tool, Observation)
to JSON log files so they can be reviewed for documentation and grading.

Each session produces one file:  logs/session_<timestamp>.json

Log entry structure
-------------------
{
  "session_id": "...",
  "turn": 1,
  "user_input": "...",
  "intermediate_steps": [
      {
          "thought": "...",
          "tool": "...",
          "tool_input": "...",
          "observation": "..."
      },
      ...
  ],
  "final_answer": "..."
}
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict


class ReasoningLogger:
    """Saves per-turn reasoning traces to a JSON file."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Unique session identifier based on timestamp
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"session_{self.session_id}.json"

        # In-memory list of all logged turns
        self.entries: List[Dict] = []

        print(f"[ReasoningLogger] Logging to {self.log_path}")

    def log_turn(
        self,
        turn: int,
        user_input: str,
        agent_output: dict,
    ) -> None:
        """
        Extract and persist one conversation turn.

        Parameters
        ----------
        turn : int
            Turn number (1-indexed).
        user_input : str
            The raw user message.
        agent_output : dict
            The full dict returned by AgentExecutor.invoke(), which includes
            'output' (final answer) and 'intermediate_steps' (list of tuples).
        """
        steps = []
        for action, observation in agent_output.get("intermediate_steps", []):
            step = {
                "tool": action.tool,
                "tool_input": action.tool_input,
                "observation": str(observation)[:2000],  # cap length
            }
            # The agent's thought lives in the log string; extract from action
            if hasattr(action, "log") and action.log:
                # Strip leading/trailing whitespace
                raw_log = action.log.strip()
                # Thought is everything before "Action:"
                thought_part = raw_log.split("Action:")[0].replace("Thought:", "").strip()
                step["thought"] = thought_part
            steps.append(step)

        entry = {
            "session_id": self.session_id,
            "turn": turn,
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "intermediate_steps": steps,
            "final_answer": agent_output.get("output", ""),
        }

        self.entries.append(entry)
        self._save()

    def _save(self) -> None:
        """Write all entries to the JSON log file (overwrite on each save)."""
        with open(self.log_path, "w", encoding="utf-8") as fh:
            json.dump(self.entries, fh, indent=2, ensure_ascii=False)

    def print_trace(self, agent_output: dict, turn: int) -> None:
        """
        Print a formatted reasoning trace to stdout for real-time visibility.

        Format
        ------
        ══ Turn N ══════════════════════
        Thought  : ...
        Tool     : ...
        Input    : ...
        Observation: ...
        ────────────────────────────────
        Final Answer: ...
        """
        separator = "═" * 60
        thin_line = "─" * 60
        print(f"\n{separator}")
        print(f"  REASONING TRACE — Turn {turn}")
        print(separator)

        steps = agent_output.get("intermediate_steps", [])
        if not steps:
            print("  (No tool calls were made — answer produced directly.)")
        for i, (action, observation) in enumerate(steps, 1):
            raw_log = getattr(action, "log", "").strip()
            thought = raw_log.split("Action:")[0].replace("Thought:", "").strip()
            print(f"\n  Step {i}")
            if thought:
                print(f"  Thought    : {thought[:400]}")
            print(f"  Tool       : {action.tool}")
            print(f"  Tool Input : {action.tool_input}")
            print(f"  Observation: {str(observation)[:500]}")

        print(f"\n{thin_line}")
        print(f"  Final Answer:\n  {agent_output.get('output', '')}")
        print(f"{separator}\n")
