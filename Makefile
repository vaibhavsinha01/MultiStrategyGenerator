install:
	pip install -r requirements.txt

run:
	python web_dashboard.py

test:
	pytest

lint:
	ruff .

format:
	black .

redis:
	redis-server

prometheus:
	C:\Prometheus\prometheus.exe

docker:
	docker compose up --build

clean:
	del /Q logs\*