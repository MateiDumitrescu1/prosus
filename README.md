
# Setup

### Files
Place the `embeddings.jsonl` file in the `tag_embeddings_voyage-3.5-lite_20251109_183615` folder.
If was too big to be stored in github.

### Backend
```bash
uv venv --python 3.11 
uv pip install -e . # install deps
# cd into backend\src\server\server.py
python server.py # start the FastAPI backend
```

### Frontend

```
npm install
npm run dev
```