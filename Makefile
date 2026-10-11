.PHONY: install api ui test lint fmt docker-up docker-down clean

install:            ## pasang dependency API + UI + dev
	pip install -r requirements-api.txt -r requirements-ui.txt -r requirements-dev.txt

api:                ## jalankan REST API (http://localhost:8000/docs)
	uvicorn api.main:app --reload --port 8000

ui:                 ## jalankan Streamlit (http://localhost:8501)
	streamlit run ui/app.py

test:               ## jalankan seluruh test + coverage
	pytest --cov=api --cov-report=term-missing

lint:               ## cek gaya kode
	ruff check api ui tests && ruff format --check api ui tests

fmt:                ## rapikan kode otomatis
	ruff format api ui tests && ruff check --fix api ui tests

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache htmlcov coverage.xml
	find . -name __pycache__ -type d -exec rm -rf {} +
