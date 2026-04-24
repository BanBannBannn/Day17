"""main.py — Lab 17 entry point.

Usage:
  python main.py                    # interactive manual REPL
  python main.py --mode auto        # auto-run all 10 benchmark conversations
  python main.py --mode benchmark   # full benchmark suite with BENCHMARK.md
  python main.py --help
"""

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)


# ─── Agent Factory ────────────────────────────────────────────────────────────

def _make_agent(session_id: str = "main_session"):
    from src.memory.manager import MemoryManager
    from src.memory.profile import UserProfile
    from src.agent.memory_agent import MemoryAgent

    episodes_dir = os.getenv("EPISODES_DIR", "./data/episodes")
    chroma_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    profiles_dir = "./data/profiles"

    Path(episodes_dir).mkdir(parents=True, exist_ok=True)
    Path(profiles_dir).mkdir(parents=True, exist_ok=True)

    mm = MemoryManager(
        session_id=session_id,
        episodes_dir=episodes_dir,
        chroma_persist_dir=chroma_dir,
    )
    profile = UserProfile(session_id=session_id, profiles_dir=profiles_dir)
    return MemoryAgent(session_id=session_id, memory_manager=mm, profile=profile)


# ─── Manual (Interactive REPL) ────────────────────────────────────────────────

def run_manual():
    """Interactive REPL with single MemoryAgent."""
    agent = _make_agent()

    print()
    print("=" * 62)
    print("  Lab 17 - Multi-Memory Agent  [MANUAL MODE]")
    print("  Backends: ShortTerm | LongTerm | Episodic | Semantic")
    print("=" * 62)
    print("Commands:")
    print("  stats          - show memory stats + profile")
    print("  reset          - clear all memory for fresh session")
    print("  knowledge <x>  - index text into semantic memory")
    print("  profile        - show current user profile facts")
    print("  episodes       - list recent episodic memories")
    print("  exit / quit    - exit")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if user_input.lower() == "stats":
            stats = agent.get_stats()
            mem = stats["memory"]
            print(f"\n[Memory Stats]")
            print(f"  Session   : {stats['session_id']}")
            print(f"  Turns     : {stats['turns_completed']}")
            print(f"  Short-term: {mem['short_term_messages']} pairs")
            print(f"  Long-term : {mem['long_term_entries']} entries")
            print(f"  Episodic  : {mem['episodic_episodes']} episodes")
            print(f"  Semantic  : {mem['semantic_documents']} documents")
            print(f"  Profile   : {stats['profile_facts']}\n")
            continue

        if user_input.lower() == "profile":
            facts = agent.profile.get_all()
            if facts:
                print("\n[User Profile]")
                for k, v in sorted(facts.items()):
                    print(f"  {k}: {v}")
                conflicts = agent.profile.conflict_history()
                if conflicts:
                    print(f"  [conflicts resolved: {len(conflicts)}]")
            else:
                print("  (no profile facts stored yet)")
            print()
            continue

        if user_input.lower() == "episodes":
            episodes = agent.memory.episodic.get_recent(n=5)
            if episodes:
                print("\n[Recent Episodes]")
                for ep in episodes:
                    ts = time.strftime("%H:%M:%S", time.localtime(ep.get("timestamp", 0)))
                    print(f"  [{ts}] ({ep['event_type']}) {ep['summary']}")
            else:
                print("  (no episodes recorded yet)")
            print()
            continue

        if user_input.lower() == "reset":
            agent.reset_session()
            print("[All memory cleared - fresh session]\n")
            continue

        if user_input.lower().startswith("knowledge "):
            text = user_input[len("knowledge "):].strip()
            if text:
                doc_id = agent.add_knowledge(text)
                print(f"[Indexed to semantic memory - doc_id: {doc_id[:8]}...]\n")
            continue

        result = agent.chat(user_input)
        profile_now = result.get("profile", {})
        budget = result.get("token_budget", {})

        print(f"\nAgent: {result['response']}")
        print(
            f"  [intent={result['intent']} | backends={result['backends_used']} | "
            f"profile_facts={len(profile_now)} | "
            f"tokens={budget.get('total_tokens','?')}/{budget.get('max_tokens','?')} | "
            f"{result['latency_ms']:.0f}ms]\n"
        )


# ─── Auto Mode ────────────────────────────────────────────────────────────────

def run_auto():
    """Auto-run all 10 benchmark conversations and print turn-by-turn results."""
    from benchmarks.conversations import BENCHMARK_CONVERSATIONS
    from src.memory.manager import MemoryManager
    from src.memory.profile import UserProfile
    from src.agent.memory_agent import MemoryAgent

    print()
    print("=" * 62)
    print("  Lab 17 - Multi-Memory Agent  [AUTO MODE]")
    print("  Running all 10 benchmark conversations...")
    print("=" * 62)

    for conv in BENCHMARK_CONVERSATIONS:
        print(f"\n{'='*62}")
        print(f"  [{conv['id']}] {conv['group']}")
        print(f"  Topic: {conv['topic']}")
        print(f"{'='*62}")

        # Fresh agent per conversation
        tmpdir = tempfile.mkdtemp()
        mm = MemoryManager(
            session_id=conv["id"],
            episodes_dir=tmpdir + "/ep",
            chroma_persist_dir=tmpdir + "/ch",
        )
        mm.long_term._use_fallback = True
        profile = UserProfile(session_id=conv["id"], profiles_dir=tmpdir + "/pr")
        agent = MemoryAgent(session_id=conv["id"], memory_manager=mm, profile=profile)

        # Pre-load semantic knowledge if any
        for chunk in conv.get("knowledge", []):
            agent.add_knowledge(chunk)
            print(f"  [Semantic knowledge indexed: {chunk[:60]}...]")

        print()
        for idx, turn in enumerate(conv["turns"]):
            human = turn["human"]
            expected = turn.get("expected_keywords", [])

            result = agent.chat(human)
            response = result["response"]

            # Keyword hit check
            hits = sum(1 for kw in expected if kw.lower() in response.lower())
            relevance = hits / len(expected) if expected else 1.0
            status = "PASS" if relevance >= 0.3 else "PARTIAL"

            print(f"  Turn {idx+1}/{len(conv['turns'])}")
            print(f"  You  : {human}")
            print(f"  Agent: {response}")
            print(f"  [intent={result['intent']} | {status} rel={relevance:.2f} | "
                  f"profile={len(result.get('profile',{}))} facts | "
                  f"{result['latency_ms']:.0f}ms]")
            print()

        # Final profile state
        final_profile = agent.profile.get_all()
        conflicts = agent.profile.conflict_history()
        print(f"  Final Profile: {final_profile}")
        if conflicts:
            for c in conflicts:
                print(f"  [Conflict resolved] {c['key']}: '{c['old_value']}' -> '{c['new_value']}'")
        print()

    print("=" * 62)
    print("  Auto mode complete.")
    print("  Run 'python main.py --mode benchmark' to generate BENCHMARK.md")
    print("=" * 62)


# ─── Benchmark Mode ───────────────────────────────────────────────────────────

def run_benchmark():
    """Full benchmark suite producing BENCHMARK.md."""
    from benchmarks.benchmark_suite import BenchmarkSuite
    suite = BenchmarkSuite()
    suite.run()


# ─── Entry Point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Lab 17 - Advanced Memory Systems for AI Agents"
    )
    parser.add_argument(
        "--mode",
        choices=["manual", "auto", "benchmark"],
        default="manual",
        help=(
            "manual  : interactive REPL (default)\n"
            "auto    : auto-run all 10 conversations and print results\n"
            "benchmark: full benchmark suite + generate BENCHMARK.md"
        ),
    )
    args = parser.parse_args()

    if args.mode == "manual":
        run_manual()
    elif args.mode == "auto":
        run_auto()
    elif args.mode == "benchmark":
        run_benchmark()


if __name__ == "__main__":
    main()
