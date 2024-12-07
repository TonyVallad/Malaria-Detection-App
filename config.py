class Config:
    # Secret Key
    SECRET_KEY = 'your_secret_key'
    
    # ANSI escape codes for colored output
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"  # White

    # Specify the directory where the training data will be stored
    DATA_DIR = 'src/static/data'
    
    # AI Model
    MODEL_DIR = "src/static/model/"
    MODEL_NAME = 'model.keras'
    MODEL_PATH = MODEL_DIR + MODEL_NAME