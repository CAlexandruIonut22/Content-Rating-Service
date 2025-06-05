import os

class Config:
    # Setări generale
    SECRET_KEY = os.environ.get('SECRET_KEY', '1234')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///ratings.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Unde salvez fișierele încărcate
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB limita
    
    # Ce tipuri de fișiere accepta
    # TODO: sa adaug si alte formate posibil
    ALLOWED_EXTENSIONS = {
        'text': ['txt', 'pdf', 'doc', 'docx'],  # pdf ptc este des folosit
        'video': ['mp4', 'mov', 'avi'],         # mov pentru Mac users
        'audio': ['mp3', 'wav', 'ogg']          # ogg -> open source
    }
    
    # Setări pentru LLM
    LLM_MODEL_ID = os.environ.get('LLM_MODEL_ID', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    LLM_CACHE_DIR = os.environ.get('LLM_CACHE_DIR', './model_cache')
    LLM_USE_4BIT = False  # Nu e nevoie pentru TinyLlama