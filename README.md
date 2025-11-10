
# Setup

### Files
Place the `embeddings.jsonl` file in the `tag_embeddings_voyage-3.5-lite_20251109_183615` folder.
If was too big to be stored in github.
The file can be found at: https://drive.google.com/file/d/1Wq95B0ON2mXfSw7TBiDOWesATyX3Jgyc/view?usp=sharing.

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

# Presentation

Excalidraw diagram: https://link.excalidraw.com/l/2ZcMrPTn79L/65TwIf5dT3p
Google slides: https://docs.google.com/presentation/d/1rZZ8MI--K4-NVf-MLFNS6d0geA-VQcu2RTxFwqkLtvc/edit?usp=sharing