PYTHON ?= python
VENV ?= .venv

install:
	$(PYTHON) -m pip install -r requirements.txt

run-demo:
	$(PYTHON) -m yes_forecast_risk.cli run-demo

test:
	pytest -q

api:
	uvicorn yes_forecast_risk.api.main:app --host 0.0.0.0 --port 8000
