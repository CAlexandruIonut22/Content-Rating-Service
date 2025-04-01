# app/utils.py
import os
from werkzeug.utils import secure_filename
from datetime import datetime
import validators
from app.config import Config
import json

def check_file_type(filename):
    """Verifică dacă tipul fișierului este permis și determină categoria conținutului."""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    
    for content_type, extensions in Config.ALLOWED_EXTENSIONS.items():
        if ext in extensions:
            return content_type
    
    return False

def save_uploaded_file(file):
    """Salvează un fișier încărcat cu un nume unic."""
    filename = secure_filename(file.filename)
    
    # Adaugă timestamp pentru evitarea coliziunilor de nume
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    new_filename = f"{timestamp}_{filename}"
    
    # Calea completă unde va fi salvat fișierul
    file_path = os.path.join(Config.UPLOAD_FOLDER, new_filename)
    
    # Salvează fișierul
    file.save(file_path)
    return file_path

def is_valid_url(url):
    """Verifică dacă un URL este valid."""
    return validators.url(url)

def store_ai_analysis(content, analysis_result):
    """Stochează rezultatele analizei AI în baza de date."""
    if not analysis_result:
        return
    
    try:
        content.ai_factuality_score = analysis_result.get('factuality_score', 0)
        content.ai_analysis_data = json.dumps(analysis_result)
        return True
    except Exception as e:
        print(f"Eroare la stocarea analizei AI: {str(e)}")
        return False

def get_ai_analysis(content):
    """Recuperează analiza AI stocată pentru conținut."""
    if not content.ai_analysis_data:
        return None
    
    try:
        return json.loads(content.ai_analysis_data)
    except:
        return None