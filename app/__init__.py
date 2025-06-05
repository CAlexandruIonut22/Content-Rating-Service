# app/__init__.py - versiunea corectată pentru encoding
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import os
import logging
import sys

# Setez encoding-ul implicit pentru a evita problemele cu caracterele românești
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

# Configurez logging-ul cu encoding UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log", encoding='utf-8'),  # IMPORTANT: encoding UTF-8
        logging.StreamHandler()
    ]
)

# Inițializez Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'parola-super-secreta-nu-spune-nimanui')  # Eliminat ă
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///ratings.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # scoate warning-urile enervante
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB, ca tot primesc PDF-uri mari

# Verific daca exista directorul de upload, daca nu, il creez
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Inițializez baza de date
db = SQLAlchemy(app)

# Import modele
from app import models

# Creez tabelele - trebuie sa fie dupa import models
with app.app_context():
    db.create_all()

# Import rute - dupa ce am creat tabelele
from app import routes