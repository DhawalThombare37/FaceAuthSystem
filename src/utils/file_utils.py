import pandas as pd
import os
from datetime import datetime

LOG_FILE = "logs/logs.csv"
STATE_FILE = "logs/state.csv"

def log_event(user, action):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([[user, action, now]])

    os.makedirs("logs", exist_ok=True)

    if os.path.exists(LOG_FILE):
        row.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        row.to_csv(LOG_FILE, header=["user", "action", "timestamp"], index=False)

    print(f"✅ Logged {action} for {user} at {now}")

def load_states():
    if os.path.exists(STATE_FILE):
        df = pd.read_csv(STATE_FILE)
        return dict(zip(df["user"], df["state"]))
    return {}

def save_states(states):
    df = pd.DataFrame(list(states.items()), columns=["user", "state"])
    df.to_csv(STATE_FILE, index=False)
