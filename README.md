## Requirement Extraction Pipeline (LangChain + OpenRouter/Groq)

This project ingests a set of business/LLM-spec documents, chunks them, summarizes each chunk in parallel, synthesizes a corpus summary, discovers relationships between documents, and finally generates a set of evaluation criteria for your LLM.

### Features
- Parallel chunk summarization with LangChain
- Choose between OpenRouter or Groq Cloud models
- Optional Gemini (Google Generative AI) support
- Ingestion for `.txt`, `.md`, `.pdf`, `.docx`
- Corpus-level synthesis, relationship discovery, and criteria generation
- Simple CLI runner that outputs a single `output.json`

### Load-Balanced Summarization Examples
- `LB.py` defines a reusable `LoadBalancer` that can:
  - Spin up clients from model names (e.g., Groq + OpenRouter routes).
  - Or wrap already-instantiated LangChain clients via `LoadBalancer.from_models(...)`.
- `lb_summarizer.py` shows chunk criteria extraction running in parallel with the load balancer plus an `asyncio.Semaphore` to keep a fixed number of concurrent calls.
- `lb_from_models_demo.py` demonstrates mixing Groq, OpenRouter, and Gemini clients that were created manually, then running prompts through the balancer in parallel.

Run either demo after exporting the required keys (e.g., `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`). For example:

```powershell
$env:GROQ_API_KEY="gsk_..."
$env:OPENROUTER_API_KEY="sk-or-..."
$env:GOOGLE_API_KEY="ya29..."
python lb_summarizer.py
# or
python lb_from_models_demo.py
```

Both scripts print which prompt was handled along with the model response, illustrating how workloads can be fanned out across providers safely.

## FastAPI Server

Start the API server (dev):

```bash
uvicorn server:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Process documents by uploading exactly one `spec_file` and zero or more `other_files`:

```bash
curl -X POST http://localhost:8000/process ^
  -F "spec_file=@path\to\LLM_spec.docx" ^
  -F "other_files=@path\to\business_doc1.pdf" ^
  -F "other_files=@path\to\business_doc2.md"
```

Notes:
- The server prefixes the spec filename with `SPEC__` internally for traceability.
- The endpoint returns JSON containing chunk summaries, corpus summary, relationship analysis, and evaluation criteria.

### Setup
1. Create a virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:
- Use `LLM_PROVIDER` to switch providers: `openrouter` or `groq`
- Set a default model with `LLM_MODEL` or rely on the defaults
- Provide your API key(s):
  - OpenRouter: `OPENROUTER_API_KEY`
  - Groq: `GROQ_API_KEY`

Example (PowerShell):

```powershell
$env:LLM_PROVIDER="openrouter"
$env:OPENROUTER_API_KEY="sk-or-..."
# Optional:
$env:LLM_MODEL="openrouter/auto"
```

Or for Groq:
$env:LLM_PROVIDER="gemini"
$env:GOOGLE_API_KEY="your_google_api_key"
$env:LLM_MODEL="gemini-1.5-pro-latest"


```powershell
$env:LLM_PROVIDER="groq"
$env:GROQ_API_KEY="gsk_..."
$env:LLM_MODEL="llama-3.1-70b-versatile"
```

You can also place these in a `.env` file at the project root; `python-dotenv` will load it automatically.

### Usage

```bash
python main.py <input_path> --out output.json
```

- `input_path` can be a file or directory. All supported files in a directory are recursively loaded.
- Optional flags:
  - `--chunk-size` (default: 1500)
  - `--chunk-overlap` (default: 200)
  - `--out` (default: `output.json`)

The output JSON contains:
- `chunk_summaries`: list of per-chunk summaries with source and position
- `corpus_summary`: synthesized executive summary
- `relationships`: structured relationship analysis (overlaps, dependencies, contradictions, traceability)
- `criteria`: JSON array of evaluation criteria suitable for LLM evaluation plans

### Notes
- OpenRouter uses an OpenAI-compatible client. We set `base_url=https://openrouter.ai/api/v1`.
- Groq uses the `langchain-groq` integration.
- Gemini uses `langchain-google-genai` and requires `GOOGLE_API_KEY`.
- Adjust models, temperatures, and prompts in `config.py`, `summarization.py`, `relationships.py`, and `criteria.py` to fit your needs.


