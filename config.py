class Config:
    # Secret Key
    SECRET_KEY = 'your_secret_key'
    
    # Image settings
    IMG_SIZE = 128
    BATCH_SIZE = 32
    
    # AI Model
    MODEL_DIR = "src/static/model/"
    MODEL_NAME = 'model.keras'
    MODEL_PATH = MODEL_DIR + MODEL_NAME
    
    # COLOR SCHEME -------------------------------------------------------------------------------
    
    # ANSI escape codes for colored output
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    RESET = "\033[0m"  # White
    
    # Color Profile 1 - Red and Purple on Black background
    NOT_INF_COLOR = "#AD1838"
    # NOT_INF_COLOR = "#C60C31"  # Red from template website
    # NOT_INF_COLOR = "#4CAF50"  # Green
    # NOT_INF_COLOR = "#504CAF"  # Blue
    INF_COLOR = "#9A54B3"
    # BG_COLOR = "#050505"
    BG_COLOR = "#050505"
    TOP_BANNER_BG_COLOR = BG_COLOR
    
    # Color Profile 2
    NOT_INF_COLOR = "#AD1838"  # Old green
    INF_COLOR = "#9A54B3"
    BG_COLOR = "#050505"
    TOP_BANNER_BG_COLOR = BG_COLOR