# Hand Gesture Controller

Control your computer with hand gestures using a webcam, MediaPipe, and a Groq LLM decision engine.

---

## Architecture

```
Webcam → MediaPipe (gesture.py) → Groq Agent (agent.py) → OS Actions (actions.py)
                                       ↕ cache / fallback
                FastAPI server (main.py) ← frontend polls /status every 500ms
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 – 3.12 (3.11 recommended) |
| Webcam | Any USB or built-in camera |
| OS | Windows 10/11 or Linux |

**Linux extra dependencies** (install via package manager, not pip):
```bash
sudo apt install xdotool pulseaudio-utils   # Ubuntu / Debian
```

---

## Setup

### 1. Clone / copy the project

```bash
cd gesture-controller
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and add your Groq API key:
```
GROQ_API_KEY=gsk_...
```

Get a free key at https://console.groq.com/keys

> **Note:** If `GROQ_API_KEY` is not set, the system falls back to the hardcoded rule table — it still works, just without LLM-powered gesture mapping.

---

## Running

```bash
cd backend
python main.py
```

The server starts on `http://localhost:8000`.

Open the frontend in a browser:

```
frontend/index.html
```

(Open the file directly or serve it with `python -m http.server 3000` from the `frontend/` directory.)

---

## Usage

1. Open the frontend in your browser.
2. Click **Start** — the webcam activates and gesture detection begins.
3. Select a **Mode** with the mode buttons:
   - **Auto** — Groq decides the action based on gesture + active app
   - **Volume** — swipe gestures control system volume
   - **Slides** — swipes advance/reverse presentation slides
   - **Cursor** — INDEX_UP gesture moves the mouse cursor
4. Watch the **Action Log** for real-time feedback.
5. Click **Stop** to release the webcam.

---

## Supported Gestures

| Gesture | Description | Default Action |
|---|---|---|
| `OPEN_PALM` | All 5 fingers extended | Pause / Play (Space) |
| `FIST` | All fingers curled | Left click |
| `PINCH` | Thumb + index tips close | Volume adjust |
| `INDEX_UP` | Only index finger extended | Move cursor |
| `SWIPE_RIGHT` | Open palm moving right | Next tab / slide / track |
| `SWIPE_LEFT` | Open palm moving left | Prev tab / slide / track |

Gestures below **75% confidence** are ignored to prevent false triggers.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/status` | GET | Current gesture, app, action, FPS, mode |
| `/start` | POST | Start webcam and gesture detection |
| `/stop` | POST | Stop webcam and threads |
| `/mode/{mode}` | POST | Switch mode: `auto`, `volume`, `slides`, `cursor` |
| `/logs` | GET | Last 50 action log entries |

---

## Running Tests

```bash
cd backend
python -m pytest test_gesture.py test_agent.py test_actions.py test_api.py -v
```

Individual test files:

```bash
python -m pytest test_gesture.py -v   # Classifier unit tests
python -m pytest test_agent.py -v     # Agent / cache / fallback tests
python -m pytest test_actions.py -v   # Action executor tests
python -m pytest test_api.py -v       # FastAPI endpoint tests
```

---

## Project Structure

```
gesture-controller/
├── .env.example           # API key template
├── README.md
├── backend/
│   ├── main.py            # FastAPI server + shared state
│   ├── gesture.py         # MediaPipe hand landmark classifier
│   ├── agent.py           # Groq LLM agent with LRU cache + fallback
│   ├── actions.py         # OS action executor (volume, keyboard, cursor)
│   ├── requirements.txt
│   ├── test_gesture.py
│   ├── test_agent.py
│   ├── test_actions.py
│   └── test_api.py
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```

---

## Troubleshooting

**No webcam found**
- The server starts anyway; `/status` returns `webcam_active: false`.
- Check `cv2.VideoCapture(0)` index — try index `1` if you have multiple cameras.

**mediapipe import error on Python 3.13**
- Use Python 3.11: `pyenv install 3.11.9 && pyenv local 3.11.9`

**pycaw / volume not working on Windows**
- Run the terminal as Administrator for COM audio API access.

**xdotool not found on Linux**
- `sudo apt install xdotool` — app detection degrades gracefully to "other" if missing.

**Groq API timeout**
- The agent falls back to hardcoded rules automatically within 2 seconds.
- Check your `GROQ_API_KEY` in `.env`.

---

## Graceful Shutdown

Press `Ctrl+C` in the terminal. The server signals all background threads to stop,
releases the webcam, and exits cleanly.
