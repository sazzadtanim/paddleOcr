# PaddleOCR API (English, forced)

Minimal FastAPI wrapper around PaddleOCR, forced to English (`lang="en"`).
Built for deployment on Dokploy via Docker Compose.

## Deploy on Dokploy

1. Push this repo to GitHub/GitLab (or any git remote Dokploy can pull from).
2. In Dokploy: create a new **Compose** service, point it at this repo.
3. Dokploy will build the image from `Dockerfile` and run `docker-compose.yml`.
4. Assign a domain to the `paddleocr` service in Dokploy's UI (it listens on
   internal port `8000`, not published to host — Dokploy's Traefik handles
   routing/SSL).
5. First request after deploy will be slow (models download into the
   `paddleocr_models` volume). Subsequent restarts reuse the cached volume.

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
- Adjust `deploy.resources.limits.memory` in `docker-compose.yml` based on
  your VPS size.
