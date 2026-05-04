# VeriTrip

This repository contains the code for an AI-powered travel planning system that uses large language models (LLMs) with retrieval-augmented search, vision-language capabilities, and multi-dimensional evaluation of generated travel plans.

## Directory Structure

```
.
├── searcher/                  # Retrieval infrastructure
│   ├── searchers/             # Pluggable backends: BM25, FAISS, ReasonIR
│   ├── mcp_server.py          # MCP server (stdio/SSE/streamable-http)
│   ├── search_r1_server.py    # FastAPI REST server (POST /retrieve)
│   ├── tools.py               # FastMCP tool registration
│   ├── restaurant.py          # City-based restaurant data + fuzzy search
│   ├── image_searcher.py      # DINOv2 + FAISS visual similarity search
│   └── ...
├── search_agent_w_img_retriver/  # LLM agent orchestration
│   ├── tool_handler.py        # Combined tool handler (search + restaurant + image)
│   ├── prompts.py             # System prompts + output format template
│   ├── openai_client_vl_retriver.py   # OpenAI VL client with tool-use loop
│   ├── claude_client_vl_retriver.py   # Anthropic Claude VL client
│   ├── gemini_client_vl_retriver.py   # Google Gemini VL client
│   └── ...                    # Other provider clients
└── evaluation/                # Evaluation framework
    ├── eval.py   # Main evaluation entry point (CLI)
    ├── trustworth.py              # Factual validation against ground truth
    └── route_eval.py        # TSP-based route optimality metric
```

## Data and Model Preparation

Set the following environment variables or edit the hardcoded paths in the source files directly.

### Datasets

```bash
export DATA_PATH="/path/to/datasets"
# Expected structure:
# ${DATA_PATH}/OpenTrip_ready/all_csv/
#   ├── {city}_poi_en.csv          # POI database per city
#   ├── {city}_{route}_trans_en.csv      # Ground truth transportation
#   ├── {city}_{route}_accom_en.csv      # Ground truth accommodation
#   └── {city}_{route}_attractions_en.csv # Ground truth attractions
```

### Model Checkpoints

```bash
export EMBEDDING_MODEL_PATH="/path/to/Qwen3-Embedding-8B"
export DINOv2_CHECKPOINT_PATH="/path/to/dinov2_vits14_pretrain.pth"
export DINOv2_SOURCE_DIR="/path/to/dinov2_source_code"
```

Models can be obtained from HuggingFace or the official project repositories.

## Usage

### 1. Start the Search Service

**MCP server:**
```bash
python searcher/mcp_server.py \
    --searcher-type faiss \
    --index-path "${INDEX_PATH}/*.faiss" \
    --model-name "${EMBEDDING_MODEL_PATH}" \
    --dataset-name "${DATA_PATH}/corpus.parquet" \
    --transport sse \
    --port 8000 \
    --k 10 \
    --snippet-max-tokens 512 \
    --get-document
```

**REST server (alternative):**
```bash
python searcher/search_r1_server.py \
    --searcher-type bm25 \
    --index-path "${BM25_INDEX_PATH}" \
    --port 8001 \
    --k 10
```

### 2. Run the Travel Planning Agent

```bash
python search_agent_w_img_retriver/openai_client_vl_retriver.py \
    --searcher-type faiss \
    --index-path "${INDEX_PATH}/*.faiss" \
    --model-name "${EMBEDDING_MODEL_PATH}" \
    --dataset-name "${DATA_PATH}/corpus.parquet" \
    --query "${DATA_PATH}/queries.csv" \
    --model "gpt-4o" \
    --output-dir "./results" \
    --k 10 \
    --snippet-max-tokens 100 \
    --max-iterations 50
```

Replace `openai_client_vl_retriver.py` with the appropriate client file (`claude_client_vl_retriver.py`, `gemini_client_vl_retriver.py`, etc.) for different LLM providers.

**Example query input format** (CSV or plain text):
```
Day 1-3, New York City, $5000, 2 adults
Must-see attractions: Statue of Liberty, Central Park
Preferences: vegetarian-friendly, near subway
```

### 3. Evaluate Generated Plans

```bash
python evaluation/eval.py \
    --query_path "${DATA_PATH}/queries.csv" \
    --txt_file "./results/" \
    --save_dir "./eval_results/"
```

The evaluation produces three metrics:
- **Commonsense score**: budget compliance, timing, transport continuity, etc.
- **Trustworthiness score**: factual accuracy against ground truth (transport IDs, hotel prices ±10%, attraction opening hours)
- **Route optimality (Average Margin)**: TSP-based metric comparing planned routes vs. optimal routes

## Output Format

The LLM agent generates structured travel plans with three tables:
- `transportationTable`: flights, trains, with specific trip IDs and times
- `accomdationTable`: hotel name, room type, price per night, breakfast info
- `itineraryTable`: daily schedule with POIs, visit durations, and meals

See `search_agent_w_img_retriver/prompts.py` (`QUERY_TEMPLATE`) for the exact format specification.

## Architecture Overview

The system follows a three-stage pipeline:

1. **Retrieval** (`searcher/`): Pluggable search backends (BM25, FAISS dense vectors, ReasonIR) served via MCP or REST API. Supports token-based snippet truncation and document retrieval by ID.

2. **Agent Orchestration** (`search_agent_w_img_retriver/`): Multi-provider LLM clients implement a tool-use loop — the agent can search for information, look up restaurants, and retrieve relevant images, feeding tool outputs back into the conversation until a complete plan is formed.

3. **Evaluation** (`evaluation/`): Three-pass evaluation comparing LLM output against ground truth data on commonsense constraints, factual correctness, and route efficiency (TSP-based).
