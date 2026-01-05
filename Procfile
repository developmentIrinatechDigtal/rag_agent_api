web: uvicorn app:app --host 0.0.0.0 --port $PORT --workers 2
dev: python -u app.py
venv: bash -lc ". .venv/bin/activate && python -u app.py"
docker: docker compose up --build -d
logs: docker compose logs --tail=200 hse-api
healthcheck: curl http://127.0.0.1:5000/health
