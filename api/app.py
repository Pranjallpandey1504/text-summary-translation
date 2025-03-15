import os
from flask import Flask, render_template, request, redirect, url_for, send_file
from transformers import pipeline
from deep_translator import GoogleTranslator
import torch
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Vercel-compatible configuration
UPLOAD_FOLDER = '/tmp/uploads'  # Only writable location
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Force CPU usage
device = "cpu"
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
translator = GoogleTranslator(source="auto", target="kn")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:
                # Read the text from the uploaded file
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()

                # Summarize the text
                summary = summarizer(text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]

                # Translate full text and summary to Kannada
                full_kannada = translator.translate(text)
                summary_kannada = translator.translate(summary)

                # Save translations to files
                full_trans_path = os.path.join(UPLOAD_FOLDER, "full_translation.txt")
                summary_trans_path = os.path.join(UPLOAD_FOLDER, "summary_translation.txt")

                with open(full_trans_path, "w", encoding="utf-8") as f:
                    f.write(full_kannada)
                with open(summary_trans_path, "w", encoding="utf-8") as f:
                    f.write(summary_kannada)

                # Display download links
                return render_template("index.html", 
                    full_trans_path="download/full_translation.txt",
                    summary_trans_path="download/summary_translation.txt"
                )

            except Exception as e:
                return f"Error processing file: {str(e)}"

            finally:
                # Clean up uploaded file after processing
                if os.path.exists(filepath):
                    os.remove(filepath)

    return render_template("index.html")

@app.route("/download/<filename>")
def download(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(filepath, as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
