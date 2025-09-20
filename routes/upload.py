import os
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from resume_parser.parsers import parse_resume
from resume_parser.utils import extract_image_from_pdf

upload_bp = Blueprint("upload", __name__)

@upload_bp.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        f = request.files["file"]
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Invalid file type"}), 400

        filename = secure_filename(f.filename)
        file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
        f.save(file_path)

        # Extract fields (PyResparser → Regex fallback → Normalization)
        fields = parse_resume(file_path)

        # Image extraction
        image_data = extract_image_from_pdf(file_path)

        return jsonify({
            "success": True,
            "fields": fields,
            "has_image": bool(image_data),
            "image": image_data,
            "manual_image_required": not bool(image_data),
        })

    except Exception as e:
        current_app.logger.exception("Upload failed")
        return jsonify({"error": f"An error occurred: {e}"}), 500
