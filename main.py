"""
main.py
-------
Healthcare Information Guide Agent — Interactive CLI Entry Point

Usage
-----
  python main.py

Prerequisites
-------------
  1. Copy .env.example to .env and fill in OPENAI_API_KEY.
  2. Install dependencies: pip install -r requirements.txt
  3. Run: python main.py

How it works
------------
  • The ReAct agent processes each user message using multi-step reasoning.
  • Reasoning traces (Thought → Tool → Observation) are printed live and
    saved to logs/session_<timestamp>.json after every turn.
  • The agent remembers previous symptoms/questions within a session
    via ConversationBufferMemory.
  • Type 'quit' or 'exit' or press Ctrl-C to end the session.
"""

import os
import sys
from dotenv import load_dotenv

# Load .env file (must happen before importing agent which reads GROQ_API_KEY)
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print(
        "\n[ERROR] GROQ_API_KEY is not set.\n"
        "  1. Copy .env.example to .env\n"
        "  2. Add your Groq API key\n"
        "  3. Re-run: python main.py\n"
    )
    sys.exit(1)

from agent import build_agent
from reasoning_logger import ReasoningLogger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          🏥  Healthcare Information Guide Agent  🏥          ║
╠══════════════════════════════════════════════════════════════╣
║  I can help you understand general healthcare information    ║
║  based on your symptoms.                                     ║
║                                                              ║
║  ⚠  DISCLAIMER: This system is NOT a medical diagnostic     ║
║     tool. Always consult a qualified healthcare professional ║
║     for personal medical advice.                             ║
╠══════════════════════════════════════════════════════════════╣
║  Type 'quit' or 'exit' to end the session.                  ║
╚══════════════════════════════════════════════════════════════╝
"""

EXAMPLES = """
💡 Example questions you can ask:
   • "I have a fever and a headache. What could it be?"
   • "What should I do if I have a sore throat?"
   • "When should I see a doctor for chest pain?"
   • "What are common flu symptoms and home remedies?"
"""


def main() -> None:
    """Run the interactive Healthcare AI Agent CLI."""
    print(BANNER)
    print(EXAMPLES)

    # Initialise the agent and the reasoning logger
    print("[System] Initialising agent …")
    agent = build_agent(verbose=True)
    logger = ReasoningLogger(log_dir="logs")
    print("[System] Agent ready.\n")

    turn = 0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n[System] Session ended by user. Goodbye! 👋")
            break

        # Exit commands
        if user_input.lower() in {"quit", "exit", "bye", "q"}:
            print("\n[System] Thank you for using the Healthcare Information Guide Agent.")
            print("[System] Remember: always consult a healthcare professional for personal advice.")
            break

        if not user_input:
            print("  (Please type a message or 'quit' to exit.)")
            continue

        turn += 1
        print(f"\n[Agent — Turn {turn}] Thinking …\n")

        try:
            # Run the agent — returns dict with 'output' and 'intermediate_steps'
            result = agent.invoke({"input": user_input})

            # Pretty-print reasoning trace to terminal
            logger.print_trace(result, turn)

            # Save reasoning trace to JSON log
            logger.log_turn(turn, user_input, result)

            # Final answer is also printed inside print_trace, but repeat it
            # clearly labelled for the user
            print(f"Agent: {result.get('output', 'No response generated.')}\n")

        except Exception as exc:
            error_msg = (
                f"[Error] The agent encountered a problem: {exc}\n"
                "Please try rephrasing your question."
            )
            print(error_msg)
            # Log the error turn too
            logger.log_turn(turn, user_input, {"output": error_msg, "intermediate_steps": []})


if __name__ == "__main__":
    main()
