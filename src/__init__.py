from config import Config
from flask import Flask
from src.routes.routes import main

def create_app():
    """
    Initializes and configures the Flask application.

    :return: The configured Flask application instance.
    """
    # app = Flask(__name__)
    app = Flask(__name__, static_folder='static')
    app.config.from_object(Config)
    
    app.secret_key = Config.SECRET_KEY

    # Register blueprints
    app.register_blueprint(main)

    return app