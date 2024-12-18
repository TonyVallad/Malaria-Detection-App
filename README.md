**<h1 align="center">Malaria Detection App</h1>**

<p align="center">
  <!-- <img src="src/static/img/logo.png" alt="Nutri-Score Logo"> -->
  <img src="src/static/img/cells.webp" alt="Nutri-Score Logo" width="300">
</p>

A web application that allows users to upload files for malaria detection. The app processes uploaded images or files to assist in the detection of malaria, providing a simple and user-friendly interface.

## Table of Contents
1. [Features](#features)
2. [Training Data](#training-data)
3. [Technologies Used](#technologies-used)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Screenshots](#screenshots)
7. [Usage](#usage)
8. [Project Structure](#project-structure)
9. [Environment Variables](#environment-variables)
10. [Future Enhancements](#future-enhancements)
11. [Contributing](#contributing)
12. [License](#license)
13. [Acknowledgments](#acknowledgments)

---

## Features
- **File Upload:** Upload images for analysis.
- **Responsive Design:** Accessible on desktop and mobile devices.
- **Simple Interface:** User-friendly and easy to navigate.

## Training Data
- **Website:** https://www.tensorflow.org/datasets/catalog/malaria?hl=fr
- **Samples:**

<p align="center">
  <img src="src/static/img/samples.png" alt="Samples">
</p>

---

## Technologies Used
- **Backend:** [Flask](https://flask.palletsprojects.com/)
- **Frontend:** HTML, CSS, Bootstrap (for responsive design)
- **Language:** Python

---

## Prerequisites
Make sure you have the following installed:
- Python 3.8 or later
- pip (Python package installer)

---

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/TonyVallad/Malaria-Detection-App.git
   cd Malaria-Detection-App
   ```

2. **Set Up a Virtual Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate       # For MacOS/Linux
   .venv\Scripts\activate          # For Windows
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Screenshots
<p align="center">
  <img src="src/static/img/screenshot_index.png" alt="Index Screenshot">
</p>

<p align="center">
  <img src="src/static/img/screenshot_infected.png" alt="Infected Screenshot">
</p>

<p align="center">
  <img src="src/static/img/screenshot_not_infected.png" alt="Not Infected Screenshot">
</p>

## Usage

1. **Run the Flask Development Server:**
   ```bash
   python app.py
   ```
   The server will start on `http://127.0.0.1:8000`.

2. **Open the Application:**
   Open your browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```

3. **Upload Files:**
   - Drop the image file or click on the dropzone to select image to analyze.
   - Press the 'Analyze' button.
   - The server will process the file and display relevant information or results.

---

## Project Structure

```
Malaria-Detection-App/
│
├── src/                    # Source code folder
│   ├── modules/            # Contains Python modules for creating and training the model
│   │   └── model_create.py
│   ├── routes/             # Contains route definitions for the Flask app
│   │   └── routes.py
│   ├── static/             # Static files for the application
│   │   ├── img/            # Images used in the app
│   │   ├── model/          # Directory containing the trained model
│   │   │   └── model.keras
│   │   └── uploads/        # Directory to store uploaded files (created automatically)
│   └── templates/          # HTML templates for the application
│       ├── base.html       # Base template (top banner)
│       ├── index.html      # Landing page template
│       └── analysis.html   # Analysis results page
│
├── .gitignore              # Git ignore file
├── app.py                  # Main application entry point
├── config.py               # Application configuration settings
├── LICENSE                 # License file
├── Model_Creation.ipynb    # Jupyter notebook for model creation and testing
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies
```

---

## Environment Variables

This app uses a secret key for session management. Set the following environment variable in your development or production environment:

- **`SECRET_KEY`**
  - Example:
    ```bash
    export SECRET_KEY="your_secret_key"
    ```

---

## Future Enhancements

With the upload functionality and machine learning model already operational, the next steps for improving the Malaria Detection App are as follows:

### Planned Improvements:
1. **UI Enhancements:**
   - Improve the user interface to make it more visually appealing and user-friendly.

2. **Enhanced Analysis Results:**
   - Provide clearer and more detailed analysis results, including additional insights where possible.

3. **Training Data Information:**
   - Develop a `/training-data` route to display detailed information about the dataset used to train the model.

4. **Model Details:**
   - Create a `/model-details` route to display technical information about the machine learning model, including architecture, performance metrics, and training details.

5. **Analysis History:**
   - Implement a `/history` route to show all previous image analyses and their results.
   - Use a SQLite database to store analysis results, enabling users to review past analyses.

### Potential Additions (Under Consideration):
- **Image Validation and Preprocessing:**
  - Develop a model to verify whether an uploaded image is a valid cell image and properly prepared.
  - Ensure that irrelevant parts of the image (non-cell pixels) are painted black before analysis.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Acknowledgments

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Bootstrap Documentation](https://getbootstrap.com/)