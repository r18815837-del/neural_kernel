# Neural Kernel

AI-powered coding assistant — custom ML framework + FastAPI backend + VS Code extension.

A learning project built from scratch in Python. No PyTorch, no TensorFlow — just NumPy (and optionally CuPy for GPU).

## Architecture

```
neural_kernel/
│
├── ML Core ─────────────────────────────────────────────
│   Deep learning framework from scratch.
│   Autograd, layers, optimizers, BPE tokenizer, training.
│   ├── kernel/          — Core ML engine
│   ├── models/          — Model architectures (LM, tokenizers)
│   ├── benchmarks/      — ML performance benchmarks
│   ├── checkpoints/     — Saved weights
│   ├── data/            — Training corpus
│   ├── tests_cuda/      — GPU tests
│   └── tests_parity/    — Numerical parity tests
│
├── API Backend ─────────────────────────────────────────
│   FastAPI server — code analysis, fix, test generation, LLM integration.
│   ├── api/             — Routes, auth, middleware, config
│   ├── cognition/       — CodingSpecialist, CodeExecutor
│   ├── llm/             — LLM clients (Anthropic, OpenAI, Ollama)
│   ├── persistence/     — SQLite store, sessions, versioning
│   ├── nk_app/          — AssistantManager orchestration
│   ├── integration/     — Client contract service
│   ├── scripts/         — Utility scripts
│   └── tests/           — API and integration tests
│
├── VS Code Extension ───────────────────────────────────
│   Sidebar AI assistant with 17 commands.
│   Explain, Fix, Generate Tests, Debug, Review, Search, Multi-Edit.
│   ├── integrations/vscode/extension.js  — Entry point
│   ├── integrations/vscode/chatView.js   — Webview UI
│   ├── integrations/vscode/src/          — Modules (commands, context, debug, review)
│   └── integrations/vscode/benchmarks/   — 18 eval fixtures + runner
│
├── Flutter Client ──────────────────────────────────────
│   └── flutter_client/  — Mobile client (Android/iOS)
│
└── Other ───────────────────────────────────────────────
    ├── company_agents/  — Multi-agent workflows
    ├── execution/       — Code execution sandbox
    ├── runtime/         — Runtime specs and text processing
    ├── examples/        — Example scripts
    └── docs/            — Documentation
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure LLM
cp .env.example .env
# Edit .env: set NK_LLM_PROVIDER=anthropic and ANTHROPIC_API_KEY

# 3. Start API
python -m uvicorn api.server:app --reload

# 4. Run benchmarks
node integrations/vscode/benchmarks/runBenchmarks.js --verbose

# 5. VS Code extension — open integrations/vscode/ in VS Code, press F5
```

## Per-module details

Each major module has its own README:

- [ML Core](kernel/README.md) — deep learning framework
- [API Backend](api/README.md) — FastAPI server + LLM integration
- [VS Code Extension](integrations/vscode/README.md) — sidebar assistant + benchmarks

## Version

- Extension: **v0.9.4**
- Backend: **v0.9.4** (with LLM integration)
- Next: **v1.0.0-beta** (docs, demo, packaging)
