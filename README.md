# PaddleOCR API (English, forced)

Minimal FastAPI wrapper around PaddleOCR, forced to English (`lang="en"`) and
running **PP-OCRv6** (paddleocr 3.7.0 / paddlepaddle 3.1.1). Built for
deployment on Dokploy via Docker Compose.

## Deploy on Dokploy

1. Push this repo to GitHub/GitLab (or any git remote Dokploy can pull from).
2. In Dokploy: create a new **Compose** service, point it at this repo.
3. Dokploy will build the image from `Dockerfile` and run `docker-compose.yml`.
4. Assign a domain to the `paddleocr` service in Dokploy's UI (it listens on
   internal port `8000`, not published to host — Dokploy's Traefik handles
   routing/SSL).
5. The PP-OCRv6 models (~150MB) are **downloaded during the Docker build**
   and baked into the image, so the first request after deploy is fast and the
   container needs no runtime network access. Each image rebuild re-downloads
   them once (build needs network).

## Endpoints

- `GET /health` → `{"status": "ok", "lang": "en"}`
- `POST /ocr` (multipart form, field `file`) → recognized text + boxes

### Example

```bash
curl -X POST -F "file=@test.jpg" https://your-domain.com/ocr
```

Response:

```json
{
  "results": [
    {"text": "Hello World", "confidence": 0.98, "box": [[...]]}
  ]
}
```

## Notes

- Language is hardcoded to English in `app.py` — not configurable via env var.
- Model version is **PP-OCRv6** (the default for paddleocr 3.x). To pin a
  different version/tier, edit the `ocr_version` / `text_*_model_name` args in
  `app.py` (and mirror the change in the Dockerfile pre-download step).
- Adjust `deploy.resources.limits.memory` in `docker-compose.yml` based on
  your VPS size (2G is comfortable for the PP-OCRv6 medium tier on CPU).
