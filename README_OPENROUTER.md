# LangChain Academy — OpenRouter Port

This is a port of the official [langchain-ai/langchain-academy](https://github.com/langchain-ai/langchain-academy) course (Introduction to LangGraph, modules 0-6) to run on **OpenRouter** instead of the OpenAI API directly, using OpenRouter's OpenAI-compatible interface.

Companion repo for the other LangChain Academy course: [Introduction-to-LangChain-academy.langchain](https://github.com/theduodecim/Introduction-to-LangChain-academy.langchain).

## What changed vs. the original repo

- Every `ChatOpenAI(...)` call across the notebooks and `studio/*.py` files now reads its model name from an env var (`OPENROUTER_MODEL`) instead of a hardcoded `"gpt-4o"` / `"gpt-3.5-turbo-0125"`.
- Every place that sets `OPENAI_API_KEY` now also sets `OPENAI_BASE_URL=https://openrouter.ai/api/v1`, so `langchain-openai` talks to OpenRouter transparently.
- Each `module-x/studio/.env.example` and the root `.env.example` now include `OPENAI_BASE_URL`, `OPENROUTER_MODEL`, and `OPENROUTER_FALLBACK_MODEL`.
- Nothing else changed: same modules, same notebooks, same `studio/` graphs, same final structure as upstream.

## Required accounts and environment variables

```
OPENAI_API_KEY=your_openrouter_api_key      # get one at https://openrouter.ai/keys
OPENAI_BASE_URL=https://openrouter.ai/api/v1

OPENROUTER_MODEL=nvidia/nemotron-3.5-lightning:free
OPENROUTER_FALLBACK_MODEL=meta-llama/llama-3.3-70b-instruct:free

TAVILY_API_KEY=your_tavily_api_key           # needed for module-4 (web search) notebooks

LANGSMITH_API_KEY=your_langsmith_api_key     # optional, recommended for tracing
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=langchain-academy-openrouter
```

Notes:

- OpenRouter exposes an OpenAI-compatible API, so `langchain-openai`'s `ChatOpenAI` class works unchanged — only the base URL and model name differ.
- Default model is `nvidia/nemotron-3.5-lightning:free` (released Aug 11, 2026): a 30B MoE model with 3B active params, 1M token context, built for agentic/tool-calling workloads. It's brand new, so if you hit tool-calling issues or availability hiccups on a specific notebook, switch to the fallback by setting `OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free` in your `.env` (no code changes needed anywhere).
- Both models above are on OpenRouter's free tier, which has rate limits (per-minute and daily). If a notebook throws a 429, wait a bit or switch models.

## Setup

Same as the original repo, plus the env vars above:

```
python3 -m venv lc-academy-env
source lc-academy-env/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env with your keys
jupyter notebook
```

## Using Studio per module

Each `module-x/studio/` folder needs its own `.env` (Studio reads it via `langgraph.json`'s `"env": "./.env"`):

```
for i in {1..5}; do
  cp module-$i/studio/.env.example module-$i/studio/.env
done
```

Then edit each `module-x/studio/.env` with your real `OPENAI_API_KEY` (your OpenRouter key). `OPENAI_BASE_URL`, `OPENROUTER_MODEL`, and `OPENROUTER_FALLBACK_MODEL` are already filled in with defaults.

```
cd module-1/studio
langgraph dev
```

## Known caveats

- `nvidia/nemotron-3.5-lightning:free` launched Aug 11, 2026 — currently served by a single provider on OpenRouter, so availability may be less stable than more established free models for the first few weeks.
- Tool-calling support for this specific model through the OpenAI-compatible interface hasn't been battle-tested across every notebook here yet — if `module-1/agent.ipynb`, `module-4/research-assistant.ipynb`, or any other tool-calling notebook misbehaves, switch to `OPENROUTER_FALLBACK_MODEL`.
- Free-tier OpenRouter models have rate limits; long agent loops (multi-agent, deep research) may hit them faster than single calls.
- `module-6` deployment notebooks/docs are unchanged conceptually — just make sure the deployed environment's `.env` also carries `OPENAI_BASE_URL` and `OPENROUTER_MODEL`.
