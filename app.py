import os

from flask import Flask, jsonify, render_template, send_from_directory

import detector

app = Flask(__name__)

DETECTIONS_PATH = detector.OUTPUT_JSON


def ensure_detections():
    if not os.path.exists(DETECTIONS_PATH):
        detector.analyze_video()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detections")
def api_detections():
    ensure_detections()
    with open(DETECTIONS_PATH, "r", encoding="utf-8") as f:
        return app.response_class(f.read(), mimetype="application/json")


@app.route("/api/reprocess", methods=["POST"])
def api_reprocess():
    data = detector.analyze_video()
    return jsonify({"ok": True, "events": len(data["events"])})


if __name__ == "__main__":
    ensure_detections()
    app.run(debug=True, port=5000, use_reloader=False)
