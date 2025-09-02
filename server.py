from flask import Flask, request, jsonify
from mindwatch import MindWatchAnalyzer
import os

app = Flask(__name__)

# Initialize analyzer with your model path
MODEL_PATH = "emotion_best.pt"  # update path to your model
analyzer = MindWatchAnalyzer(MODEL_PATH)

@app.route("/health")
def health():
    return jsonify({"status": "Backend is running"})

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    # Run analysis
    if file.filename.lower().endswith((".jpg", ".png", ".jpeg")):
        analyzer.process_single_image(filepath, output_path="outputs/result.jpg")
    elif file.filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        analyzer.process_video(filepath, output_path="outputs/result.mp4")
    else:
        return jsonify({"error": "Unsupported file type"}), 400

    # Save report
    report_path = analyzer.save_detailed_report("outputs/report.json")

    with open(report_path, "r") as f:
        report = f.read()

    return report

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    app.run(debug=True)
