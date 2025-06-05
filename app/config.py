import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Setări generale
    SECRET_KEY = os.environ.get('SECRET_KEY', '1234')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///ratings.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Unde salvez fișierele încărcate
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB limita
    
    # Ce tipuri de fișiere accepta
    ALLOWED_EXTENSIONS = {
        'text': ['txt', 'pdf', 'doc', 'docx'],
        'video': ['mp4', 'mov', 'avi'],
        'audio': ['mp3', 'wav', 'ogg']
    }
    
    # ===== SETĂRI PENTRU MODELUL LLM UPGRADED =====
    
    # ALEGE MODELUL (decomentează doar unul):
    
    # OPȚIUNEA 1: Mistral 7B (RECOMANDAT - fără restricții, excelent pentru factualitate)
    # LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
    # LLM_USE_4BIT = True  # Economie RAM pentru Mistral
    
    # OPȚIUNEA 2: Llama 3.1 8B (necesită token Hugging Face)
    LLM_MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
    LLM_USE_4BIT = True
    
    # OPȚIUNEA 3: TinyLlama (model mic pentru teste rapide)
    # LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # LLM_USE_4BIT = False
    
    # OPȚIUNEA 4: Qwen 2.5 (alternativă bună)
    # LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    # LLM_USE_4BIT = True
    
    # Setări comune pentru toate modelele
    LLM_CACHE_DIR = os.environ.get('LLM_CACHE_DIR', './model_cache')
    
    # ===== SETĂRI HARDWARE =====
    FORCE_GPU = True  # True = doar pe GPU, False = permite CPU
    GPU_MEMORY_FRACTION = 0.9  # Folosește 90% din memoria GPU
    MULTI_GPU = False  # True dacă ai mai multe GPU-uri
    
    # ===== AUTENTIFICARE HUGGING FACE =====
    # Necesară doar pentru modele restricționate (Llama 3.1)
    HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN', None)
    # SAU setează direct aici token-ul tău:
    # HUGGINGFACE_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxx"
    
    # ===== WEB SEARCH (pentru funcționalități viitoare) =====
    TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY', None)
    USE_WEB_SEARCH = os.environ.get('USE_WEB_SEARCH', 'false').lower() == 'true'
    
    # ===== SETĂRI DE DEBUG =====
    DEBUG_LLM = os.environ.get('DEBUG_LLM', 'false').lower() == 'true'