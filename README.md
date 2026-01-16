# Clinical Guideline Chat

## Backend (FastAPI)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

Environment variables (in `.env`):
- `WEAVIATE_URL`
- `WEAVIATE_API_KEY`
- `OPENAI_API_KEY`
- `OPENAI_MODEL` (default: `gpt-5.2`)
- `FOLLOWUP_MODEL` (default: `gpt-5.2`)
- `LANGSMITH_API_KEY` (optional)
- `LANGSMITH_PROJECT` (optional)
- `LANGSMITH_PROMPT` (optional)

## Frontend (Next.js)

```bash
cd frontend
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_BASE` if the API is not on `http://localhost:8000`.

Open `http://localhost:3000` in the browser.
