STREAMLIT_APP = ui.py
FASTAPI_APP = service

.PHONY: frontend backend run_app

frontend:
	/Users/dinayatsuk/PycharmProjects/Calorie\ Tracker/.venv/bin/streamlit run $(STREAMLIT_APP)

backend:
	/Users/dinayatsuk/PycharmProjects/Calorie\ Tracker/.venv/bin/uvicorn $(FASTAPI_APP):app &

run_app: backend frontend