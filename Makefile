.PHONY: serve ingest test

serve:
	uvicorn src.api.main:app --reload

ingest:
	python -m src.cli ingest $(REPO)

test:
	pytest tests/
