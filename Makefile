.PHONY: serve ingest test eval up

serve:
	uvicorn src.api.main:app --reload

ingest:
	python -m src.cli ingest $(REPO)

test:
	pytest tests/

eval:
	python -m evals.runner

up:
	docker compose up --build
