"""10 multi-turn benchmark conversations covering all rubric test groups.

Groups required by rubric:
  G1 — Profile recall
  G2 — Conflict update
  G3 — Episodic recall
  G4 — Semantic retrieval
  G5 — Trim / token budget
"""

from typing import Dict, List

BENCHMARK_CONVERSATIONS: List[Dict] = [

    # ── G1: Profile Recall ─────────────────────────────────────────────────

    {
        "id": "conv_01",
        "group": "G1_profile_recall",
        "topic": "Name recall after multiple turns",
        "description": (
            "User states their name early. After 5 filler turns, agent must "
            "recall it from long-term profile without re-asking."
        ),
        "turns": [
            {
                "human": "My name is Linh. Nice to meet you!",
                "expected_keywords": ["noted", "profile", "remember", "Linh", "got it"],
                "expected_no_memory": "I don't know your name",
            },
            {"human": "What is Python?", "expected_keywords": ["python", "language", "programming"]},
            {"human": "Tell me about machine learning.", "expected_keywords": ["machine learning", "data", "model"]},
            {"human": "Explain neural networks briefly.", "expected_keywords": ["neural", "layers", "network"]},
            {"human": "What is Docker used for?", "expected_keywords": ["docker", "container"]},
            {"human": "What is my name?", "expected_keywords": ["Linh"], "expected_no_memory": "don't know"},
        ],
    },

    {
        "id": "conv_02",
        "group": "G1_profile_recall",
        "topic": "Job and language preference recall",
        "description": "Agent stores job and language preference, recalls them later.",
        "turns": [
            {
                "human": "I work as a data scientist and I prefer Python.",
                "expected_keywords": ["noted", "profile", "data scientist", "python", "got it"],
                "expected_no_memory": "I don't have that information",
            },
            {"human": "What are the best Python libraries for data science?",
             "expected_keywords": ["pandas", "numpy", "scikit", "tensorflow"]},
            {"human": "How does gradient descent work?",
             "expected_keywords": ["gradient", "loss", "minimize", "learning rate"]},
            {
                "human": "What is my job?",
                "expected_keywords": ["data scientist"],
                "expected_no_memory": "don't know",
            },
        ],
    },

    # ── G2: Conflict Update ─────────────────────────────────────────────────

    {
        "id": "conv_03",
        "group": "G2_conflict_update",
        "topic": "Allergy conflict correction (rubric mandatory test)",
        "description": (
            "User first says they are allergic to milk. Then corrects to soy. "
            "Profile must show only soy. No-memory agent retains the wrong fact."
        ),
        "turns": [
            {
                "human": "I am allergic to milk.",
                "expected_keywords": ["noted", "milk", "profile", "allergy", "got it"],
                "expected_no_memory": "ok",
            },
            {"human": "What foods should I avoid?",
             "expected_keywords": ["milk", "dairy", "allergy"]},
            {
                "human": "Actually, I am allergic to soy, not milk.",
                "expected_keywords": ["updated", "soy", "corrected", "resolved", "conflict"],
                "expected_no_memory": "ok",
            },
            {
                "human": "What am I allergic to?",
                "expected_keywords": ["soy"],
                "expected_no_memory": "milk",  # wrong — baseline keeps stale fact
            },
        ],
    },

    {
        "id": "conv_04",
        "group": "G2_conflict_update",
        "topic": "Location correction",
        "description": "User first says they live in Hanoi, then corrects to Ho Chi Minh City.",
        "turns": [
            {
                "human": "I am from Hanoi.",
                "expected_keywords": ["noted", "hanoi", "profile", "got it"],
            },
            {"human": "What are popular coffee shops in Hanoi?",
             "expected_keywords": ["hanoi", "coffee", "cafe"]},
            {
                "human": "Actually I moved. I live in Ho Chi Minh City now.",
                "expected_keywords": ["updated", "ho chi minh", "corrected", "resolved"],
            },
            {
                "human": "Where am I from?",
                "expected_keywords": ["ho chi minh"],
                "expected_no_memory": "hanoi",
            },
        ],
    },

    # ── G3: Episodic Recall ─────────────────────────────────────────────────

    {
        "id": "conv_05",
        "group": "G3_episodic_recall",
        "topic": "Debug lesson recall",
        "description": (
            "User completes a debug task using docker service names. "
            "Later asks agent to recall the solution."
        ),
        "turns": [
            {
                "human": "I just fixed a bug: my container was connecting to 'localhost' but it should use the Docker service name 'db'.",
                "expected_keywords": ["noted", "docker", "service", "episode", "got it"],
            },
            {"human": "What other Docker networking tips do you know?",
             "expected_keywords": ["network", "bridge", "service", "container"]},
            {"human": "Tell me about Docker Compose.",
             "expected_keywords": ["compose", "services", "yaml", "docker"]},
            {
                "human": "Remember the Docker bug I fixed earlier? What was the solution?",
                "expected_keywords": ["docker", "service name", "db", "localhost"],
                "expected_no_memory": "don't recall",
            },
        ],
    },

    {
        "id": "conv_06",
        "group": "G3_episodic_recall",
        "topic": "Project decision recall",
        "description": "Team decided to use microservices. Agent recalls this decision later.",
        "turns": [
            {
                "human": "We decided to use microservices for our new project.",
                "expected_keywords": ["noted", "microservices", "episode", "got it"],
            },
            {"human": "What are the pros and cons of microservices?",
             "expected_keywords": ["scalability", "complexity", "independent", "service"]},
            {"human": "How do we handle service-to-service authentication?",
             "expected_keywords": ["JWT", "OAuth", "token", "auth", "service"]},
            {
                "human": "What architecture did we choose for the project?",
                "expected_keywords": ["microservices"],
                "expected_no_memory": "don't know",
            },
        ],
    },

    # ── G4: Semantic Retrieval ──────────────────────────────────────────────

    {
        "id": "conv_07",
        "group": "G4_semantic_retrieval",
        "topic": "FAQ chunk retrieval — LangChain basics",
        "description": (
            "Agent is given a knowledge fragment about LangChain. "
            "Later query must retrieve and use it."
        ),
        "knowledge": [
            "LangChain is an open-source framework for building LLM applications. "
            "It provides abstractions for chains, agents, memory, and tool use. "
            "Key components: LLMs, PromptTemplates, OutputParsers, Chains, Agents.",
        ],
        "turns": [
            {"human": "What is LangChain?",
             "expected_keywords": ["langchain", "framework", "llm", "chain", "agent", "semantic"]},
            {"human": "How does LangGraph extend LangChain?",
             "expected_keywords": ["langgraph", "state", "graph", "workflow"]},
            {"human": "What is a PromptTemplate?",
             "expected_keywords": ["prompt", "template", "variable", "format"]},
            {
                "human": "Give me the key components of LangChain.",
                "expected_keywords": ["llm", "chain", "agent", "prompttemplate", "memory"],
                "expected_no_memory": "I don't have specific details",
            },
        ],
    },

    {
        "id": "conv_08",
        "group": "G4_semantic_retrieval",
        "topic": "FAQ chunk retrieval — RAG and vector stores",
        "description": "Knowledge about RAG is indexed. Agent retrieves it semantically.",
        "knowledge": [
            "RAG (Retrieval-Augmented Generation) combines a retriever with an LLM generator. "
            "The retriever fetches relevant document chunks from a vector store (Chroma, FAISS, Pinecone). "
            "The generator uses these chunks as context to produce grounded, accurate answers. "
            "RAG reduces hallucination by anchoring responses to retrieved facts.",
        ],
        "turns": [
            {"human": "What is RAG?",
             "expected_keywords": ["retrieval", "augmented", "generation", "vector", "semantic"]},
            {"human": "What vector databases work with LangChain?",
             "expected_keywords": ["chroma", "faiss", "pinecone", "vector", "store"]},
            {"human": "How does RAG reduce hallucination?",
             "expected_keywords": ["anchor", "retrieved", "facts", "grounded", "context"]},
            {
                "human": "Explain how RAG works step by step.",
                "expected_keywords": ["retriever", "vector", "generator", "chunks", "context"],
                "expected_no_memory": "I don't have details",
            },
        ],
    },

    # ── G5: Trim / Token Budget ─────────────────────────────────────────────

    {
        "id": "conv_09",
        "group": "G5_trim_token_budget",
        "topic": "Context window management under long conversation",
        "description": (
            "10-turn conversation to verify context window stays within budget "
            "and important info (profile, system prompt) is not evicted."
        ),
        "turns": [
            {"human": "My name is Nam and I am a software engineer.",
             "expected_keywords": ["noted", "nam", "engineer", "profile"]},
            {"human": "What is the CAP theorem?",
             "expected_keywords": ["consistency", "availability", "partition"]},
            {"human": "Explain eventual consistency.",
             "expected_keywords": ["eventual", "consistency", "replicas", "nodes"]},
            {"human": "What is a distributed transaction?",
             "expected_keywords": ["distributed", "transaction", "ACID", "saga"]},
            {"human": "How does Kafka work?",
             "expected_keywords": ["kafka", "topic", "partition", "producer", "consumer"]},
            {"human": "What is the difference between SQL and NoSQL?",
             "expected_keywords": ["sql", "nosql", "schema", "flexible"]},
            {"human": "Tell me about Kubernetes.",
             "expected_keywords": ["kubernetes", "pod", "cluster", "orchestration"]},
            {"human": "What is gRPC?",
             "expected_keywords": ["grpc", "protocol", "rpc", "protobuf"]},
            {
                "human": "What is my name?",
                "expected_keywords": ["nam"],
                "expected_no_memory": "don't know",
            },
            {
                "human": "How many tokens are left in the context window?",
                "expected_keywords": ["token", "budget", "remaining", "window"],
            },
        ],
    },

    {
        "id": "conv_10",
        "group": "G5_trim_token_budget",
        "topic": "Mixed scenario: profile + semantic + episodic under budget",
        "description": (
            "Combines profile recall, semantic retrieval, and episodic recall "
            "in a single conversation to test that all backends work together "
            "within the token budget."
        ),
        "knowledge": [
            "AWS Lambda is a serverless compute service that runs code in response to events. "
            "It automatically scales and charges only for execution time. "
            "Use cases: API backends, file processing, event-driven automation.",
        ],
        "turns": [
            {"human": "My name is Tung and I work at an AWS startup.",
             "expected_keywords": ["noted", "tung", "aws", "profile"]},
            {"human": "We deployed our first Lambda function today!",
             "expected_keywords": ["noted", "lambda", "episode", "great"]},
            {"human": "What is AWS Lambda?",
             "expected_keywords": ["lambda", "serverless", "aws", "event", "semantic"]},
            {"human": "What are the cold start issues with Lambda?",
             "expected_keywords": ["cold start", "latency", "warm", "lambda"]},
            {
                "human": "Remember when I told you about deploying Lambda? What did I do?",
                "expected_keywords": ["lambda", "deployed", "episode", "function"],
                "expected_no_memory": "don't recall",
            },
            {
                "human": "What is my name and where do I work?",
                "expected_keywords": ["tung", "aws"],
                "expected_no_memory": "don't know",
            },
        ],
    },
]
