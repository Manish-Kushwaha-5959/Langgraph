# LangGraph Workflows — A Hands-On Cookbook

A collection of Jupyter notebooks that explore the three core workflow patterns in **LangGraph**, from single-node LLM calls to parallel evaluators and conditional branching. Every notebook is small, self-contained, and runs end-to-end against an open-source LLM hosted on Hugging Face.

> **Model used:** `meta-llama/Llama-3.1-8B-Instruct` (via `langchain-huggingface`).
> **Stack:** `langgraph 1.2.5`, `langchain 1.3.9`, `langchain-core 1.4.7`, `pydantic 2.13`.

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
└── 03_ConditionalWorkflow/
    └── quadratic_equation.ipynb      # Quadratic solver with conditional edges
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

   The notebooks load credentials via `python-dotenv`. Create a `.env` file in the repo root:

   ```env
   HUGGINGFACEHUB_API_TOKEN=hf_your_token_here
   ```

3. **Launch Jupyter**

   ```bash
   jupyter notebook
   ```

   Open any `.ipynb` under the three numbered folders and run the cells top-to-bottom.

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

## 🧠 Key Concepts Demonstrated

| Concept | Where it appears |
|---|---|
| `StateGraph` + `TypedDict` state | All notebooks |
| `START` / `END` sentinels | All notebooks |
| Linear edges | `01_*` notebooks |
| Fan-out / fan-in (parallel branches) | `02_*` notebooks |
| `Annotated[..., operator.add]` reducer for parallel updates | `UPSC_essay_worflow.ipynb` |
| `PydanticOutputParser` for structured LLM output | `UPSC_essay_worflow.ipynb` |
| `add_conditional_edges` with a router function | `quadratic_equation.ipynb` |
| `Literal` return type on routers | `quadratic_equation.ipynb` |
| LLM-backed nodes (HuggingFace) | `simple_llm_workflow`, `prompt_chaining`, `UPSC_essay` |
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
- `langsmith==0.8.16`
- `pydantic==2.13.4`
- `python-dotenv==1.2.2`
- `huggingface-hub` (transitive via `langchain-huggingface`)

A local `venv/` directory is included; `.gitignore` keeps it out of version control, along with `.env`.

---

## 📌 Suggested Reading Order

1. Start with **`simple_llm_workflow.ipynb`** to learn the absolute basics of `StateGraph`, nodes, and edges.
2. Move to **`prompt_chaining.ipynb`** to see state propagate between nodes.
3. Try **`bmi_workflow.ipynb`** to confirm the pattern works for non-LLM logic too.
4. Open **`batsman_workflow.ipynb`** for your first parallel graph.
5. Tackle **`UPSC_essay_worflow.ipynb`** to combine parallelism with structured LLM output and reducers.
6. Finish with **`quadratic_equation.ipynb`** to learn conditional branching.

By the end you'll have built a mental model for every primitive LangGraph offers at the graph-construction level.
