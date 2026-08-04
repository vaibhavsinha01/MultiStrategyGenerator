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
	C:\Prometheus\prometheus-3.13.2.windows-amd64\prometheus.exe --config.file=prometheus.yml

grafana:
	"C:\Program Files\GrafanaLabs\grafana\bin\grafana.exe"

docker:
	docker compose up --build

clean:
	del /Q logs\*
