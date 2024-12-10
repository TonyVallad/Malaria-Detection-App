class Config:
    # Secret Key
    SECRET_KEY = 'your_secret_key'
    
    # ANSI escape codes for colored output
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"  # White
    
    # Page Color Palette
    # INF_COLOR = "#ad1888"
    NOT_INF_COLOR = "#ad1838"
    INF_COLOR = "#9A54B3"
    # NOT_INF_COLOR = "#C60C31"
    
    # Image settings
    IMG_SIZE = 128
    BATCH_SIZE = 32
    
    # AI Model
    MODEL_DIR = "src/static/model/"
    MODEL_NAME = 'model.keras'
    MODEL_PATH = MODEL_DIR + MODEL_NAME