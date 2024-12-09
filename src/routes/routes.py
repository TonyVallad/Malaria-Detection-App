import os
import numpy as np
from config import Config
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from flask import Blueprint, render_template, current_app, jsonify, request, redirect, url_for, flash

main = Blueprint('main', __name__)

# Load the model once at startup
model = load_model(Config.MODEL_PATH)

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

@main.route('/analysis', methods=['POST'])
def analysis():
    if 'file' not in request.files or not request.files['file']:
        flash("No file uploaded. Please upload an image.")
        print(f"{Config.RED}No file uploaded. Please upload an image.{Config.RESET}")
        return redirect(url_for('main.index'))
    
    file = request.files['file']

    try:
        # Save the uploaded file in src/static/uploads
        file_path = os.path.join(current_app.static_folder, "uploads", file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)

        # Preprocess the image
        image = load_img(file_path, target_size=(Config.IMG_SIZE, Config.IMG_SIZE))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize to [0, 1]

        # Predict using the loaded model
        prediction = model.predict(image)
        result = "Infected" if prediction[0][0] < 0.5 else "Not Infected"

        # Generate the file URL
        file_url = url_for('static', filename=f'uploads/{file.filename}')
        
        # Calculate result and confidence
        if prediction[0][0] < 0.5:
            result = "Infected"
            confidence = (1 - prediction[0][0]) * 100  # Map prediction to 50%-100%
        else:
            result = "Not Infected"
            confidence = prediction[0][0] * 100  # Map prediction to 50%-100%

        # Render the analysis result
        return render_template('analysis.html', result=result, file_url=file_url, confidence=confidence)

    except Exception as e:
        flash(f"An error occurred: {str(e)}")
        print(f"{Config.RED}An error occurred:{Config.RESET} {str(e)}")
        return redirect(url_for('main.index'))
    
@main.route('/static-test')
def static_test():
    test_path = url_for('static', filename='img/logo.png')
    return f"Static file URL: {test_path}"

@main.route('/debug-static')
def debug_static():
    return f"Static folder: {current_app.static_folder}"