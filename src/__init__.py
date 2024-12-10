from config import Config
from flask import Flask
import math
from src.routes.routes import main

def create_app():
    """
    Initializes and configures the Flask application.

    :return: The configured Flask application instance.
    """
    app = Flask(__name__, static_folder='static')
    app.config.from_object(Config)
    
    app.secret_key = Config.SECRET_KEY

    # Make Config available in Jinja templates
    app.jinja_env.globals.update(
        Config=Config,
        pi=math.pi,
        cos=math.cos,
        sin=math.sin
    )

    # Register blueprints
    app.register_blueprint(main)

    return app