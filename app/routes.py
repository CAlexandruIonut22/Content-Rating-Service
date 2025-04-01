# app/routes.py
from flask import render_template, request, redirect, url_for, flash, session, jsonify
from app import app, db
from app.models import User, Content, Rating
from app.utils import check_file_type, save_uploaded_file, is_valid_url, store_ai_analysis, get_ai_analysis
from werkzeug.utils import secure_filename
import os
import json
from datetime import datetime

# Importă modulul LLM
from app.llm_module.factuality_checker import FactualityChecker

# Inițializează verificatorul de factualitate global
factuality_checker = None

def initialize_llm():
    """Funcție pentru inițializarea modelului LLM. Poate fi apelată din run.py"""
    global factuality_checker
    try:
        # Comentează linia următoare pentru a dezactiva temporar LLM
        # from app.llm_module.factuality_checker import FactualityChecker
        # factuality_checker = FactualityChecker()
        print("LLM dezactivat temporar pentru testare")
    except Exception as e:
        print(f"Eroare la inițializarea modelului LLM: {str(e)}")

@app.route('/')
def index():
    """Pagina principală cu conținut recent."""
    recent_content = Content.query.order_by(Content.created_at.desc()).limit(10).all()
    return render_template('index.html', contents=recent_content)

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Înregistrare utilizator nou."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Verifică dacă username-ul e deja luat
        if User.query.filter_by(username=username).first():
            flash('Acest nume de utilizator este deja folosit.')
            return redirect(url_for('register'))
        
        # Creează utilizator nou
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        
        flash('Înregistrare reușită! Te poți autentifica acum.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Autentificare utilizator."""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Caută utilizatorul
        user = User.query.filter_by(username=username).first()
        
        # Verifică dacă utilizatorul există și parola e corectă
        if user and user.password == password:  # FIXME: Trebuie hash-uite parolele!
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Autentificare reușită!')
            return redirect(url_for('index'))
        else:
            flash('Nume de utilizator sau parolă incorecte.')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Delogare utilizator."""
    session.pop('user_id', None)
    session.pop('username', None)
    flash('Te-ai delogat cu succes.')
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
def upload_content():
    """Încărcare conținut nou."""
    if 'user_id' not in session:
        flash('Trebuie să fii autentificat pentru a încărca conținut.')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        title = request.form.get('title')
        
        if not title:
            flash('Te rugăm să introduci un titlu.')
            return redirect(url_for('upload_content'))
        
        content = Content(title=title)
        
        # Verifică dacă avem un fișier sau un link
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            
            # Verifică tipul fișierului
            content_type = check_file_type(file.filename)
            if not content_type:
                flash('Tip de fișier nepermis.')
                return redirect(url_for('upload_content'))
            
            # Salvează fișierul
            file_path = save_uploaded_file(file)
            
            # Actualizează informațiile despre conținut
            content.is_file = True
            content.content_type = content_type
            content.file_path = file_path
            
        elif request.form.get('url'):
            url = request.form.get('url')
            
            # Verifică dacă URL-ul e valid
            if not is_valid_url(url):
                flash('URL invalid. Te rugăm să introduci un URL valid.')
                return redirect(url_for('upload_content'))
            
            # Actualizează informațiile despre conținut
            content.is_file = False
            content.content_type = request.form.get('content_type', 'text')
            content.url = url
            
        else:
            flash('Te rugăm să încarci un fișier sau să introduci un URL.')
            return redirect(url_for('upload_content'))
        
        # Salvează în baza de date
        db.session.add(content)
        db.session.commit()
        
        flash('Conținut încărcat cu succes!')
        return redirect(url_for('view_content', content_id=content.id))
    
    return render_template('upload.html')

@app.route('/content/<int:content_id>')
def view_content(content_id):
    """Vizualizează un conținut specific și evaluările sale."""
    content = Content.query.get_or_404(content_id)
    
    # Calculează ratingurile medii
    avg_ratings = content.get_avg_ratings()
    
    # Obține toate ratingurile pentru acest conținut
    ratings = Rating.query.filter_by(content_id=content.id).all()
    
    # Obține analiza AI dacă există
    ai_analysis = get_ai_analysis(content)
    
    return render_template(
        'content.html', 
        content=content,
        average_ratings=avg_ratings,
        ratings=ratings,
        ai_analysis=ai_analysis
    )

@app.route('/content/<int:content_id>/rate', methods=['GET', 'POST'])
def rate_content(content_id):
    """Evaluează un conținut specific."""
    if 'user_id' not in session:
        flash('Trebuie să fii autentificat pentru a evalua conținut.')
        return redirect(url_for('login'))
    
    content = Content.query.get_or_404(content_id)
    
    if request.method == 'POST':
        # Preia valorile din formular
        coherence = int(request.form.get('coherence', 5))
        truth = int(request.form.get('truth', 5))
        attractiveness = int(request.form.get('attractiveness', 5))
        comment = request.form.get('comment', '')
        
        # Asigură-te că valorile sunt între 1 și 10
        coherence = max(1, min(10, coherence))
        truth = max(1, min(10, truth))
        attractiveness = max(1, min(10, attractiveness))
        
        # Verifică dacă utilizatorul a evaluat deja acest conținut
        existing_rating = Rating.query.filter_by(
            content_id=content.id,
            user_id=session['user_id']
        ).first()
        
        if existing_rating:
            # Actualizează evaluarea existentă
            existing_rating.coherence = coherence
            existing_rating.truth = truth
            existing_rating.attractiveness = attractiveness
            existing_rating.comment = comment
            flash('Evaluarea ta a fost actualizată.')
        else:
            # Creează o evaluare nouă
            rating = Rating(
                content_id=content.id,
                user_id=session['user_id'],
                coherence=coherence,
                truth=truth,
                attractiveness=attractiveness,
                comment=comment
            )
            db.session.add(rating)
            flash('Evaluarea ta a fost înregistrată.')
        
        db.session.commit()
        return redirect(url_for('view_content', content_id=content.id))
    
    # Verifică dacă utilizatorul a evaluat deja acest conținut
    existing_rating = None
    if 'user_id' in session:
        existing_rating = Rating.query.filter_by(
            content_id=content.id,
            user_id=session['user_id']
        ).first()
    
    return render_template('rate.html', content=content, existing_rating=existing_rating)

@app.route('/browse')
def browse_content():
    """Răsfoiește conținutul cu filtrare opțională."""
    content_type = request.args.get('type')
    
    # Pornește cu toate conținuturile
    query = Content.query
    
    # Aplică filtrul dacă e specificat
    if content_type in ['text', 'video', 'audio']:
        query = query.filter_by(content_type=content_type)
    
    # Sortează după data adăugării, cel mai recent primul
    all_content = query.order_by(Content.created_at.desc()).all()
    
    return render_template('browse.html', contents=all_content)

@app.route('/api/analyze_factuality', methods=['POST'])
def analyze_factuality():
    """API endpoint pentru analiza factualității conținutului."""
    if not factuality_checker:
        return jsonify({
            "error": "Modelul LLM nu a fost inițializat corect."
        }), 500
    
    data = request.json
    if not data or 'text' not in data:
        return jsonify({
            "error": "Trebuie să furnizați textul pentru analiză."
        }), 400
    
    text = data['text']
    title = data.get('title', '')
    
    # Efectuează analiza factualității
    analysis = factuality_checker.analyze_text_content(text, title)
    
    return jsonify(analysis)

@app.route('/content/<int:content_id>/analyze', methods=['GET'])
def analyze_content(content_id):
    """Pagină pentru analiza automată a conținutului."""
    # Verifică dacă utilizatorul este autentificat
    if 'user_id' not in session:
        flash('Trebuie să fii autentificat pentru a analiza conținutul.')
        return redirect(url_for('login'))
    
    # Obține conținutul
    content = Content.query.get_or_404(content_id)
    
    # Verifică dacă avem deja o analiză salvată
    existing_analysis = get_ai_analysis(content)
    if existing_analysis:
        return render_template(
            'analyze.html', 
            content=content,
            analysis=existing_analysis
        )
    
    # Verifică dacă modelul LLM este disponibil
    if not factuality_checker:
        flash('Analiza automată nu este disponibilă momentan.')
        return redirect(url_for('view_content', content_id=content_id))
    
    # Analizează conținutul în funcție de tipul acestuia
    analysis_result = None
    
    if content.is_file and content.content_type == 'text':
        # Pentru fișiere text, încarcă și analizează conținutul
        try:
            file_path = content.file_path
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                analysis_result = factuality_checker.analyze_text_content(
                    text_content, 
                    content.title
                )
                # Salvează rezultatul analizei
                if analysis_result:
                    store_ai_analysis(content, analysis_result)
                    db.session.commit()
        except Exception as e:
            flash(f'Eroare la analiza fișierului: {str(e)}')
    elif not content.is_file:
        # Pentru link-uri, trimitem doar titlul pentru analiză
        # Notă: Într-o implementare reală, ar trebui să scraperuim conținutul de la URL
        analysis_result = {
            "factuality_score": 5,
            "confidence": 3,
            "reasoning": "Analiza link-urilor necesită extragerea conținutului de la URL, care nu este implementată în această versiune.",
            "questionable_claims": ["Nu s-a putut analiza conținutul link-ului automat."]
        }
        # Salvează rezultatul analizei
        store_ai_analysis(content, analysis_result)
        db.session.commit()
    else:
        flash('Acest tip de conținut nu poate fi analizat automat.')
    
    return render_template(
        'analyze.html',
        content=content,
        analysis=analysis_result
    )