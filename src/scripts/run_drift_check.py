import requests
import datetime

DRIFT_ALERT_URL = "http://127.0.0.1:8000/drift/alerts"
WINDOW_SIZE = 100

def run_drift_check():
    try:
        response = requests.get(
            DRIFT_ALERT_URL,
            params={"window_size": WINDOW_SIZE},
            timeout=10
        )
        print("Time:", datetime.datetime.now())
        print("Status:", response.status_code)
        print("Response:", response.text)
    except Exception as e:
        print("Drift check failed:", str(e))

if __name__ == "__main__":
    run_drift_check()