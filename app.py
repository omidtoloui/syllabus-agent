import json
import os
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# override=True so .env takes precedence over empty env vars (e.g. from Claude Desktop)
load_dotenv(override=True)

import anthropic
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)

# Load syllabus at startup
SYLLABUS_PATH = Path(__file__).parent / "syllabus.txt"
SYLLABUS_CONTENT = SYLLABUS_PATH.read_text(encoding="utf-8")

# Anthropic client
api_key = os.environ.get("ANTHROPIC_API_KEY", "")
if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
client = anthropic.Anthropic(api_key=api_key)

# Rate limiting: per-IP, 20 requests per minute
RATE_LIMIT = 20
RATE_WINDOW = 60
request_log = defaultdict(list)


def build_system_prompt():
    today = date.today().isoformat()
    return f"""You are the course assistant for MGMT 274A/B – Advanced Topics in Management \
Capstone: Advancing Health Tech Innovation at UCLA Anderson School of Management.

Today's date is {today}.

Your ONLY source of information is the course syllabus provided below. Follow these rules strictly:

1. Answer questions ONLY using information found in the syllabus below.
2. If the answer is not in the syllabus, say: "I don't see that information in the syllabus. \
You may want to reach out to the teaching team at omid.toloui@anderson.ucla.edu."
3. Do NOT speculate, guess, or provide information from outside the syllabus.
4. Do NOT give opinions about the course, the instructor, the assignments, or the companies.
5. Do NOT make jokes about or editorialize on any course content.
6. Be helpful, professional, and concise.
7. When referencing dates, assignments, or deadlines, quote them exactly as they appear in the syllabus.
8. If asked about topics not related to this course, politely redirect: \
"I can only help with questions about the MGMT 274A/B course syllabus."
9. Do NOT make up or invent any facts, statistics, URLs, or details not explicitly stated in the syllabus.
10. Keep responses focused and reasonably brief. Students can ask follow-up questions for more detail.
11. When asked about upcoming deadlines, current week, or what's next, use today's date to determine \
where the student is in the course schedule. The course runs Fall 2026 (Weeks 1-10, starting Sept 30) \
and Winter 2027 (Weeks 11-20, starting Jan 6). If the current date is before the course starts, note \
that the course hasn't begun yet and share the start date.
12. Use markdown formatting in your responses: **bold** for emphasis, bullet lists for multiple items, \
and clear structure. Keep it readable and scannable.

=== COURSE SYLLABUS ===
{SYLLABUS_CONTENT}
=== END SYLLABUS ==="""


def is_rate_limited(ip):
    now = time.time()
    request_log[ip] = [t for t in request_log[ip] if now - t < RATE_WINDOW]
    if len(request_log[ip]) >= RATE_LIMIT:
        return True
    request_log[ip].append(now)
    return False


def parse_history(data):
    """Extract and cap conversation history from request data."""
    history = data.get("history", [])
    if len(history) > 20:
        history = history[-20:]
    messages = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    return messages


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat_stream():
    """Streaming SSE endpoint — returns text chunks as they arrive from Haiku."""
    ip = request.remote_addr
    if is_rate_limited(ip):
        return jsonify({"error": "Too many requests. Please wait a moment."}), 429

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided."}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Empty message."}), 400

    messages = parse_history(data)
    messages.append({"role": "user", "content": user_message})

    def generate():
        try:
            with client.messages.stream(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=build_system_prompt(),
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'t': text})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception:
            yield f"data: {json.dumps({'error': 'Sorry, I am having trouble right now. Please try again.'})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"

    resp = Response(generate(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.route("/api/chat-sync", methods=["POST"])
def chat_sync():
    """Non-streaming fallback endpoint."""
    ip = request.remote_addr
    if is_rate_limited(ip):
        return jsonify({"error": "Too many requests. Please wait a moment."}), 429

    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided."}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Empty message."}), 400

    messages = parse_history(data)
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=build_system_prompt(),
            messages=messages,
        )
        reply = response.content[0].text
        return jsonify({"reply": reply})
    except Exception:
        return jsonify({"error": "Sorry, I'm having trouble right now. Please try again."}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)
