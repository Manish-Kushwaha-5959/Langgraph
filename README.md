# LangGraph Workflows — A Hands-On Cookbook

A collection of Jupyter notebooks and small Streamlit apps that explore the core workflow patterns in **LangGraph**, from single-node LLM calls to parallel evaluators, conditional branching, iterative refinement, persistence, and a multi-thread chatbot UI. Every example is small, self-contained, and runs end-to-end against an open-source LLM hosted on Hugging Face.

> **Model used:** `meta-llama/Llama-3.1-8B-Instruct` (via `langchain-huggingface`).
> **Stack:** `langgraph 1.2.5`, `langchain 1.3.9`, `langchain-core 1.4.7`, `pydantic 2.13`, `streamlit 1.58`.

---

## 📂 Repository Layout

```
LangGraph/
├── 01_SequentialWorkflow/
│   ├── simple_llm_workflow.ipynb     # Single-node LLM Q&A
│   ├── prompt_chaining.ipynb         # Title → Outline → Blog
│   └── bmi_workflow.ipynb            # BMI calculator + category label
├── 02_ParallelWorkflow/
│   ├── batsman_workflow.ipynb        # Parallel cricket-stats metrics
│   └── UPSC_essay_worflow.ipynb      # Parallel essay evaluator with Pydantic parser
├── 03_ConditionalWorkflow/
│   └── quadratic_equation.ipynb      # Quadratic solver with conditional edges
├── 04_IterativeWorkflow/
│   └── X_post_generator.ipynb        # Tweet generator with generate-evaluate-optimize loop
├── 05_Chatbot/
│   ├── basic_chatbot.ipynb           # Memory-backed chatbot in a notebook
│   ├── chatbot_backend.py            # Reusable compiled chatbot graph (with InMemorySaver)
│   ├── 1_chatbot_ui.py               # Streamlit chat UI (non-streaming)
│   ├── 2_chatbot_ui_streaming.py     # Streamlit chat UI with token streaming
│   ├── 3_chatbot_ui_streaming_resumechat.py  # Multi-thread UI: new chat + resume past conversations
│   └── 4_chat_test.py                # Multi-thread UI with auto-generated chat titles
└── 06_Persistence/
    └── persistence.ipynb             # Checkpointing, threads, and time-travel via get_state_history
```

---

## 🚀 Setup

1. **Create a virtual environment and install dependencies**

   ```bash
   python -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add your Hugging Face token**

   The notebooks and apps load credentials via `python-dotenv`. Create a `.env` file in the repo root:

   ```env
   HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
   ```

3. **Run a notebook**

   ```bash
   jupyter notebook
   ```

   Open any `.ipynb` under the numbered folders and run the cells top-to-bottom.

4. **Run a chatbot app (optional)**

   ```bash
   cd 05_Chatbot
   streamlit run 4_chat_test.py
   ```

---

## 1️⃣ Sequential Workflows

In a sequential graph, every node executes exactly once in a fixed order, and each node updates the shared state which the next node reads.

### `simple_llm_workflow.ipynb` — Single-Node LLM Call
The "Hello, World" of LangGraph.

- **State (`LLMState`)**: `question`, `answer`
- **Nodes**: one — `llm_qa` that prompts the model and stores its reply
- **Edges**: `START → llm_qa → END`

Demonstrates the minimum viable graph: a `TypedDict` state, one node function, and a linear edge pair.

### `prompt_chaining.ipynb` — Title → Outline → Blog
A two-step LLM chain where each step refines the previous output.

- **State (`BlogState`)**: `title`, `outline`, `content`
- **Nodes**:
  1. `create_outline` — generates a structured outline from a title
  2. `create_blog` — expands the outline into a full blog post
- **Edges**: `START → create_outline → create_blog → END`

Illustrates **prompt chaining**: the second node depends on the first node's output living in state.

### `bmi_workflow.ipynb` — Deterministic Pipeline with Categorical Labeling
A non-LLM example to show that LangGraph isn't just for language models.

- **State (`BMIState`)**: `height_m`, `weight_kg`, `bmi`, `category`
- **Nodes**:
  1. `calculate_bmi` — `weight / height²`
  2. `label_bmi` — maps the BMI to *Underweight / Normal / Overweight / Obese*
- **Edges**: `START → calculate_bmi → label_bmi → END`

---

## 2️⃣ Parallel Workflows

Parallel graphs **fan out** from a single source to multiple nodes that execute concurrently, then **fan in** to a join node before terminating. This is great for independent sub-tasks (multiple evaluators, multiple metrics).

### `batsman_workflow.ipynb` — Cricket Metrics in Parallel
Computes three independent cricket statistics from the same input, then merges them into a textual summary.

- **State (`BatsmanState`)**: `runs`, `balls`, `fours`, `sixes`, `sr`, `bpb`, `boundary_percent`, `summary`
- **Nodes**:
  - `calculate_sr` → strike rate `(runs / balls) * 100`
  - `calculate_bpb` → balls per boundary
  - `calculate_boundary_percent` → % of runs from boundaries
  - `summary` → formats all three into a string
- **Edges**: `START → {sr, bpb, boundary%} → summary → END`

A clean fan-out / fan-in with no LLM dependency — the simplest illustration of the parallel pattern.

### `UPSC_essay_worflow.ipynb` — Parallel LLM Evaluators with Structured Output
A more realistic use of parallelism: evaluate the same essay on three different dimensions at once, then synthesize a final verdict.

- **State (`essayState`)**: `essay`, `language_feedback`, `analysis_feedback`, `clarity_feedback`, `overall_feedback`, `individual_scores` (an `Annotated[list[int], operator.add]` reducer so scores from each parallel node concatenate), `avg_score`
- **Schema**: `EvaluationSchema(BaseModel)` with `feedback: str` and `score: int` (constrained `0 ≤ score ≤ 10`), parsed via `PydanticOutputParser`
- **Nodes**:
  - `evaluate_language`, `evaluate_analysis`, `evaluate_thought` — run **in parallel**, each returning its own feedback + score
  - `join_results` — a no-op placeholder that synchronizes the three branches
  - `final_evaluation` — asks the LLM to merge all three feedbacks and computes the average score
- **Edges**:
  ```
  START ─┬─▶ evaluate_language  ─┐
         ├─▶ evaluate_analysis   ─┼─▶ join_results ─▶ final_evaluation ─▶ END
         └─▶ evaluate_thought    ─┘
  ```

Highlights:
- **`Annotated[list[int], operator.add]`** — the canonical reducer pattern for combining partial updates from parallel nodes into one state field.
- **`PydanticOutputParser`** — guarantees each evaluator returns valid JSON with a score between 0 and 10.
- **Synthesis node** — uses the LLM to summarize the three feedback streams into one overall assessment.

---

## 3️⃣ Conditional Workflows

Conditional graphs use **`add_conditional_edges`** to route execution at runtime based on state. The decision function returns the **name of the next node**, enabling branching and merging.

### `quadratic_equation.ipynb` — Quadratic Equation Solver
Solves `ax² + bx + c = 0` and routes the answer based on the discriminant.

- **State (`QuadState`)**: `a`, `b`, `c`, `equation`, `discriminant`, `result`
- **Nodes**:
  1. `show_equation` — pretty-prints the equation, handling sign and zero edge cases
  2. `calculate_discriminant` — `b² − 4ac`
  3. `real_root` — two distinct real roots (`Δ > 0`)
  4. `repeated_root` — one repeated root (`Δ = 0`)
  5. `no_real_root` — complex roots (`Δ < 0`)
- **Decision function**:
  ```python
  def check_discriminant(state) -> Literal["real_root", "no_real_root", "repeated_root"]:
      if state["discriminant"] == 0: return "repeated_root"
      elif state["discriminant"] > 0: return "real_root"
      else: return "no_real_root"
  ```
- **Edges**:
  ```
  START → show_equation → calculate_discriminant ─┬─▶ real_root      ─▶ END
                                                  ├─▶ repeated_root  ─▶ END
                                                  └─▶ no_real_root   ─▶ END
  ```

The decision function's return type is a **`Literal`** — this is how LangGraph knows the possible branches statically and validates the graph at compile time.

---

## 4️⃣ Iterative Workflows

Iterative graphs **loop** over a generate → evaluate → optimize cycle, refining an artifact until it meets a quality bar or a maximum iteration count is reached. The loop terminates via a conditional edge.

### `X_post_generator.ipynb` — Tweet Generator with Self-Critique
Writes a tweet, then runs it through a critic LLM that either approves it or returns feedback. The optimizer rewrites the tweet from the feedback and the cycle repeats.

- **State (`TweetState`)**: `topic`, `tweet`, `evaluation` (`"approved" | "needs_improvement"`), `feedback`, `iteration`, `max_interation`
- **Schema**: `EvaluationSchema(BaseModel)` with `evaluation: Literal["approved", "needs_improvement"]` and `feedback: str`, parsed via `PydanticOutputParser`
- **Nodes**:
  - `generate` — produces a tweet under 500 chars in observational-humor / meme-logic style
  - `evaluate` — ruthless critic scoring originality, humor, punchiness, virality, format; auto-rejects Q&A-style jokes, oversize tweets, and deflating closers
  - `optimize` — rewrites the tweet based on the critic's feedback
- **Decision function**:
  ```python
  def route_evaluation(state):
      if state["evaluation"] == "approved" or state["iteration"] >= state["max_interation"]:
          return "approved"
      return "needs_improvement"
  ```
- **Edges**:
  ```
  START → generate → evaluate ─┬─▶ END
                                └─▶ optimize → evaluate (loop)
  ```
  Implemented with:
  ```python
  graph.add_conditional_edges(
      "evaluate",
      route_evaluation,
      {"approved": END, "needs_improvement": "optimize"},
  )
  graph.add_edge("optimize", "evaluate")
  ```

Highlights:
- **Self-critique loop** — the same LLM-as-a-judge pattern used in modern agent systems, but kept simple enough to read end-to-end.
- **`PydanticOutputParser` + `Literal`** — the critic's verdict is constrained to a strict enum, which makes the conditional edge statically checkable.
- **Bounded iteration** — the loop has a hard ceiling (`max_interation`) so a tweet that's never approved still terminates.

---

## 5️⃣ Chatbot

A progression from a notebook chatbot to a multi-threaded Streamlit app with token streaming, chat history, and auto-generated chat titles. All UIs share the same `chatbot_backend.py` — a compiled LangGraph chatbot with an `InMemorySaver` checkpointer.

### `basic_chatbot.ipynb` — Memory-Backed Chatbot in a Notebook
- **State (`ChatState`)**: `messages: Annotated[list[BaseMessage], add_messages]`
- **Nodes**: one — `chat_node` that calls the LLM with the full message history
- **Edges**: `START → chat_node → END`
- **Persistence**: `MemorySaver` checkpointer, threads keyed by `thread_id`

Highlights:
- **`add_messages` reducer** — appends new messages instead of overwriting the list, preserving the full conversation per thread.
- **Multi-turn memory** — the bot remembers the user's name and earlier answers across turns in the same thread.

### `chatbot_backend.py` — Reusable Compiled Graph
A standalone module that builds the chatbot graph, attaches an `InMemorySaver`, and exposes a compiled `chatbot` object imported by every UI. Includes a sample `chatbot.stream(..., stream_mode="messages")` call showing how to stream tokens.

### `1_chatbot_ui.py` — Streamlit Chat UI (Non-Streaming)
The simplest Streamlit front-end: stores messages in `st.session_state`, calls `chatbot.invoke(...)`, and renders the full response at once.

### `2_chatbot_ui_streaming.py` — Token Streaming
Same as above, but uses `st.write_stream` with `chatbot.stream(..., stream_mode="messages")` to render tokens as they arrive.

### `3_chatbot_ui_streaming_resumechat.py` — Multi-Thread Chat with Sidebar
Adds chat-management features on top of the streaming UI:
- A `uuid`-generated `thread_id` per conversation
- **New Chat** button to start a fresh thread
- A sidebar listing all past threads — clicking one calls `chatbot.get_state(...)` to load the saved messages and resume the conversation

### `4_chat_test.py` — Auto-Generated Chat Titles
The most complete version. On the very first user message, the LLM is asked (under a separate `"new_chat_thread"` config) to summarize the message into a 2–5 word title; the sidebar entry is then updated from the placeholder UUID to the generated title.

Highlights across the chatbot folder:
- **One backend, many UIs** — `chatbot_backend.py` is the single source of truth; UI files only differ in their presentation logic.
- **`stream_mode="messages"`** — yields `(message_chunk, metadata)` pairs so UIs can stream token-by-token.
- **Resumable threads** — `chatbot.get_state({"configurable": {"thread_id": ...}})` reconstructs any past conversation from the in-memory checkpoint store.

---

## 6️⃣ Persistence

Persistence layers give a graph **memory across invocations** by snapshotting state to a checkpointer. Each run is keyed by a `thread_id`, and LangGraph exposes APIs to inspect, replay, and time-travel the saved states.

### `persistence.ipynb` — Joke Generator with Checkpointing and Time-Travel
- **State (`JokeState`)**: `topic`, `joke`, `explanation`
- **Nodes**:
  - `gen_joke` — writes a joke for the topic
  - `exp_joke` — explains the joke
- **Edges**: `START → gen_joke → exp_joke → END`
- **Checkpointer**: `InMemorySaver()`

The notebook then demonstrates:
- **Threaded invocations** — running the same compiled graph with two different `thread_id`s (`"1"` and `"2"`) keeps each conversation independent.
- **`workflow.get_state(config)`** — returns a `StateSnapshot` for the latest checkpoint of a thread.
- **`workflow.get_state_history(config)`** — returns the full list of `StateSnapshot`s in reverse chronological order, so you can walk the graph back through every step (start → `gen_joke` → `exp_joke`).

Highlights:
- **`InMemorySaver`** — the simplest checkpointer; great for notebooks and demos. Swap it out for a database-backed one (`SqliteSaver`, `PostgresSaver`) for production.
- **Thread isolation** — same graph, different `thread_id`s, zero state leakage.
- **Time travel** — `get_state_history` lets you audit, fork, or resume from any earlier checkpoint.

---

## 🧠 Key Concepts Demonstrated

| Concept | Where it appears |
|---|---|
| `StateGraph` + `TypedDict` state | All notebooks |
| `START` / `END` sentinels | All notebooks |
| Linear edges | `01_*` notebooks |
| Fan-out / fan-in (parallel branches) | `02_*` notebooks |
| `Annotated[..., operator.add]` reducer for parallel updates | `UPSC_essay_worflow.ipynb` |
| `PydanticOutputParser` for structured LLM output | `UPSC_essay_worflow.ipynb`, `X_post_generator.ipynb` |
| `add_conditional_edges` with a router function | `quadratic_equation.ipynb`, `X_post_generator.ipynb` |
| `Literal` return type on routers | `quadratic_equation.ipynb`, `X_post_generator.ipynb` |
| Iterative generate–evaluate–optimize loop | `X_post_generator.ipynb` |
| LLM-as-a-judge with auto-reject rules | `X_post_generator.ipynb` |
| `add_messages` reducer for chat history | `basic_chatbot.ipynb`, `chatbot_backend.py` |
| `InMemorySaver` / `MemorySaver` checkpointer | `basic_chatbot.ipynb`, `chatbot_backend.py`, `persistence.ipynb` |
| Threaded conversations via `thread_id` | `05_Chatbot/*`, `persistence.ipynb` |
| Token streaming via `stream_mode="messages"` | `chatbot_backend.py`, `2/3/4_chatbot_ui*.py` |
| `get_state` / `get_state_history` (time travel) | `persistence.ipynb`, `3/4_chatbot_ui*.py` |
| Streamlit chat UI | `05_Chatbot/*.py` |
| LLM-backed nodes (HuggingFace) | `simple_llm_workflow`, `prompt_chaining`, `UPSC_essay`, `X_post_generator`, `chatbot_*` |
| Pure-computation nodes (no LLM) | `bmi_workflow`, `batsman_workflow`, `quadratic_equation` |

---

## 🛠️ Dependencies

The full pinned list lives in `requirements.txt`. Key packages:

- `langgraph==1.2.5`
- `langgraph-checkpoint==4.1.1`
- `langgraph-prebuilt==1.1.0`
- `langgraph-sdk==0.4.2`
- `langchain==1.3.9`
- `langchain-core==1.4.7`
- `langchain-huggingface==1.2.2`
- `langsmith==0.8.16`
- `pydantic==2.13.4`
- `python-dotenv==1.2.2`
- `streamlit==1.58.0`
- `huggingface-hub==1.19.0`

A local `venv/` directory is included; `.gitignore` keeps it out of version control, along with `.env` and `__pycache__`.

---

## 📌 Suggested Reading Order

1. Start with **`simple_llm_workflow.ipynb`** to learn the absolute basics of `StateGraph`, nodes, and edges.
2. Move to **`prompt_chaining.ipynb`** to see state propagate between nodes.
3. Try **`bmi_workflow.ipynb`** to confirm the pattern works for non-LLM logic too.
4. Open **`batsman_workflow.ipynb`** for your first parallel graph.
5. Tackle **`UPSC_essay_worflow.ipynb`** to combine parallelism with structured LLM output and reducers.
6. Finish **`quadratic_equation.ipynb`** to learn conditional branching.
7. Loop into **`X_post_generator.ipynb`** to see generate–evaluate–optimize in action.
8. Build memory with **`basic_chatbot.ipynb`** → `persistence.ipynb` to understand checkpointing, threads, and time travel.
9. Layer on the UI by reading `chatbot_backend.py` first, then `1_chatbot_ui.py` → `2_` → `3_` → `4_` for the Streamlit progression.

By the end you'll have built a mental model for every primitive LangGraph offers at the graph-construction level — and a working multi-thread chatbot to show for it.
