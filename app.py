from flask import Flask
import os

# --- Flask App ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Register Blueprints ---
from routes.main import main_bp
from routes.upload import upload_bp

app.register_blueprint(main_bp)
app.register_blueprint(upload_bp)

if __name__ == "__main__":
    app.run(debug=True)
