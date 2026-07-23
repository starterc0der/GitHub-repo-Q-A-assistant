.PHONY: serve ingest test up

serve:
	uvicorn src.api.main:app --reload

ingest:
	python -m src.cli ingest $(REPO)

test:
	pytest tests/

up:
	docker compose up --build
