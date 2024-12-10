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
        print("No file uploaded")
        flash("No file uploaded. Please upload an image.")
        return redirect(url_for('main.index'))

    file = request.files['file']
    print(f"File received: {file.filename}")

    try:
        # Save the uploaded file in src/static/uploads
        file_path = os.path.join(current_app.static_folder, "uploads", file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)
        print(f"File saved to: {file_path}")

        # Preprocess the image
        image = load_img(file_path, target_size=(Config.IMG_SIZE, Config.IMG_SIZE))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = image / 255.0  # Normalize to [0, 1]
        print("Image preprocessed")

        # Predict using the loaded model
        prediction = float(model.predict(image))  # Get the prediction value
        print(f"Prediction: {prediction}")

        # Determine result
        result = "Infected" if prediction < 0.5 else "Not Infected"
        print(f"Result: {result}")

        # Generate the file URL
        file_url = url_for('static', filename=f'uploads/{file.filename}')
        print(f"File URL: {file_url}")

        # Render the analysis result
        return render_template('analysis.html', result=result, file_url=file_url, prediction=prediction)

    except Exception as e:
        print(f"Error: {str(e)}")
        flash(f"An error occurred: {str(e)}")
        return redirect(url_for('main.index'))