# app/config.py
import os

class Config:
    # Configurații de bază
    SECRET_KEY = os.environ.get('SECRET_KEY', 'cheie-dezvoltare-nu-folosi-in-productie')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///ratings.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Configurații pentru upload-uri
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB max upload
    
    # Tipuri de fișiere permise
    ALLOWED_EXTENSIONS = {
        'text': ['txt', 'pdf', 'doc', 'docx'],
        'video': ['mp4', 'mov', 'avi'],
        'audio': ['mp3', 'wav', 'ogg']
    }
    
    # Configurații pentru LLM
    LLM_MODEL_ID = os.environ.get('LLM_MODEL_ID', 'mistralai/Mistral-7B-Instruct-v0.2')
    LLM_CACHE_DIR = os.environ.get('LLM_CACHE_DIR', './model_cache')
    LLM_USE_4BIT = os.environ.get('LLM_USE_4BIT', 'True').lower() == 'true'