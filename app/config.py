# app/config.py 
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Setări generale
    SECRET_KEY = os.environ.get('SECRET_KEY', '1234')
    SQLALCHEMY_DATABASE_URI = "postgresql://test_v64e_user:NY9EkSOPMPv3vAfIfXvPaNMazOVm1s4n@dpg-d12pjjruibrs73ffr9ng-a.frankfurt-postgres.render.com/test_v64e"
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
    
    # ===== SETĂRI PENTRU MODELUL LLM (OPTIMIZAT VITEZĂ) =====
    
    # TINYLLAMA - FOARTE RAPID pe orice hardware
    LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    LLM_USE_4BIT = False
    
    # Alternative pentru hardware mai puternic (decomentează dacă vrei să încerci):
    # LLM_MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"      # Prea lent pe CPU - peste 10 minute si nu a mers de la incarcare
    # LLM_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"       # Alternativă 3.8B
    # LLM_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"              # Pentru GPU/hardware puternic - not the case
    
    # Setări comune pentru toate modelele
    LLM_CACHE_DIR = os.environ.get('LLM_CACHE_DIR', './model_cache')
    
    # ===== SETĂRI HARDWARE =====
    FORCE_GPU = False  # CPU pentru compatibilitate
    GPU_MEMORY_FRACTION = 0.8
    MULTI_GPU = False
    
    # ===== AUTENTIFICARE HUGGING FACE =====
    HUGGINGFACE_TOKEN = os.environ.get('HUGGINGFACE_TOKEN', None)
    
    # ===== TAVILY WEB SEARCH (REACTIVAT) =====
    TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY', None)
    USE_WEB_SEARCH = os.environ.get('USE_WEB_SEARCH', 'true').lower() == 'true'
    
    # ===== SETĂRI ANALIZĂ HIBRIDĂ (LLM + WEB) REACTIVATE =====
    USE_HYBRID_ANALYSIS = True      # REACTIVAT - combină TinyLlama cu web search
    MAX_WEB_SEARCHES = 3           # EXACT 3 căutări cum ai cerut
    WEB_SEARCH_TIMEOUT = 10        # Timeout pentru căutări web
    
    # ===== PARAMETRI TINYLLAMA OPTIMIZAȚI PENTRU PROMPTURI MAI LUNGI =====
    LLM_DEFAULT_TEMPERATURE = 0.3    # Puțin mai mare pentru creativitate
    LLM_MAX_NEW_TOKENS = 200        # CRESCUT pentru răspunsuri mai detaliate
    LLM_CONTEXT_LENGTH = 4096       # CRESCUT pentru prompturi mai lungi
    LLM_DO_SAMPLE = True           # TRUE pentru răspunsuri mai diverse
    LLM_TOP_P = 0.9               # Pentru variație controlată
    LLM_REPETITION_PENALTY = 1.1   # Împiedică repetările
    LLM_NUM_BEAMS = 1             # Păstrează greedy pentru viteză
    
    # ===== SETĂRI DE DEBUG =====
    DEBUG_LLM = os.environ.get('DEBUG_LLM', 'true').lower() == 'true'
    DEBUG_WEB_SEARCH = False  # Dezactivat
