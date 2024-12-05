from config import Config
from flask import Blueprint, render_template, current_app, jsonify, request, redirect, url_for, flash

main = Blueprint('main', __name__)

# Home Route
@main.route('/', methods=['GET', 'POST'])
def index():
    """
    The home route of the application, which renders the index.html template.

    Returns:
        HTML: The rendered index.html template.
    """

    # Send to index.html
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    if file:
        file.save(f"uploads/{file.filename}")  # Save the file to an "uploads" directory
        flash('File successfully uploaded!')
        return redirect(url_for('index'))