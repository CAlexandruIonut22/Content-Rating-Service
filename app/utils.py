import os
from werkzeug.utils import secure_filename
from datetime import datetime
import validators
from app.config import Config
import json
import logging

logger = logging.getLogger(__name__)

def check_file_type(filename):
    """Verifică dacă fișierul e permis și returnează tipul de conținut"""
    if '.' not in filename:
        return False
    
    ext = filename.rsplit('.', 1)[1].lower()
    
    # Verifică extensia
    for content_type, extensions in Config.ALLOWED_EXTENSIONS.items():
        if ext in extensions:
            return content_type
    
    return False  # extensie nepermisă

def save_uploaded_file(file):
    """Salvează fișierul cu nume unic să nu se suprascrie"""
    filename = secure_filename(file.filename)
    
    # Adaug un timestamp - nu e cea mai elegantă soluție dar merge
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    new_filename = f"{timestamp}_{filename}"
    
    # Calea completă
    file_path = os.path.join(Config.UPLOAD_FOLDER, new_filename)
    
    # Salvez fișierul
    file.save(file_path)
    return file_path

def is_valid_url(url):
    """Verifică dacă URL-ul e valid - nu e perfectă verificarea dar e ok pentru început"""
    return validators.url(url)

def store_ai_analysis(content, analysis_result):
    """Salvează rezultatul analizei în baza de date"""
    if not analysis_result:
        return False
    
    try:
        content.ai_factuality_score = analysis_result.get('factuality_score', 0)
        content.ai_analysis_data = json.dumps(analysis_result)
        return True
    except Exception as e:
        logger.error(f"Eroare la stocarea analizei AI: {str(e)}")
        return False

def extract_text_from_file(file_path):
    """Extrage text dintr-un fișier - bazic momentan, doar pentru txt și PDF"""
    logger.info(f"Încerc să extrag text din: {file_path}")
    
    # Dacă e un fișier text simplu
    if file_path.endswith('.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"Text extras din {file_path}: {len(content)} caractere")
                return content
        except Exception as e:
            logger.error(f"Eroare la citirea fișierului text: {str(e)}")
            return ""
    
    # Pentru PDF-uri, încercăm să folosim PyPDF2
    elif file_path.endswith('.pdf'):
        try:
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            logger.info(f"Text extras din PDF {file_path}: {len(text)} caractere")
            return text
        except ImportError:
            error_msg = "PyPDF2 nu e instalat. Rulează: pip install PyPDF2"
            logger.error(error_msg)
            return f"Eroare: {error_msg}"
        except Exception as e:
            error_msg = f"Eroare la extragerea textului din PDF: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    # TODO: Adaugă suport pentru mai multe formate
    # Poate docx cu python-docx, sau alte formate
    
    return "Formatul fișierului nu este suportat pentru extragerea textului."

def extract_text_from_url(url):
    """
    Extrage text dintr-un URL folosind URL extractor-ul
    Această funcție e un wrapper pentru a fi consistentă cu stilul proiectului
    """
    try:
        from app.url_extractor import url_extractor
        logger.info(f"Extrag text din URL: {url}")
        result = url_extractor.extract_text_from_url(url)
        
        if result['success']:
            # Limitez textul pentru a evita probleme cu modelul LLM
            original_text = result['text']
            max_chars_for_analysis = 3000  # Limită pentru analiză
            
            if len(original_text) > max_chars_for_analysis:
                logger.info(f"Text din URL prea lung ({len(original_text)} caractere), limitez la {max_chars_for_analysis}")
                # Păstrez începutul textului care de obicei conține informațiile principale
                limited_text = original_text[:max_chars_for_analysis] + "..."
            else:
                limited_text = original_text
            
            logger.info(f"Text extras cu succes din {url}: {len(limited_text)} caractere (original: {len(original_text)})")
            return {
                'success': True,
                'text': limited_text,
                'title': result['title'],
                'word_count': result.get('word_count', 0),
                'char_count': len(limited_text),
                'original_char_count': len(original_text),
                'error': None
            }
        else:
            logger.warning(f"Eroare la extragerea din {url}: {result['error']}")
            return {
                'success': False,
                'text': '',
                'title': '',
                'word_count': 0,
                'char_count': 0,
                'error': result['error']
            }
    except Exception as e:
        error_msg = f"Eroare la procesarea URL-ului: {str(e)}"
        logger.error(error_msg)
        return {
            'success': False,
            'text': '',
            'title': '',
            'word_count': 0,
            'char_count': 0,
            'error': error_msg
        }

def get_content_for_analysis(content):
    """
    Obține textul pentru analiză dintr-un obiect Content
    Funcționează atât pentru fișiere cât și pentru URL-uri
    """
    if content.is_file and content.file_path:
        # Pentru fișiere
        if os.path.exists(content.file_path):
            text_content = extract_text_from_file(content.file_path)
            if text_content and not text_content.startswith("Eroare"):
                return {
                    'success': True,
                    'text': text_content,
                    'source': 'file',
                    'source_path': content.file_path
                }
            else:
                return {
                    'success': False,
                    'error': text_content if text_content else "Nu s-a putut extrage text din fișier",
                    'source': 'file'
                }
        else:
            return {
                'success': False,
                'error': "Fișierul nu există pe server",
                'source': 'file'
            }
    elif content.url:
        # Pentru URL-uri
        url_result = extract_text_from_url(content.url)
        if url_result['success']:
            return {
                'success': True,
                'text': url_result['text'],
                'source': 'url',
                'source_url': content.url,
                'title': url_result['title']
            }
        else:
            return {
                'success': False,
                'error': url_result['error'],
                'source': 'url'
            }
    else:
        return {
            'success': False,
            'error': "Conținutul nu are fișier sau URL asociat",
            'source': 'unknown'
        }