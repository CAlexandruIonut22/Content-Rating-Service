from flask import render_template, request, redirect, url_for, flash, session, jsonify
from app import app, db
from app.models import User, Content, Rating
from app.utils import check_file_type, save_uploaded_file, is_valid_url, store_ai_analysis, get_content_for_analysis
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Variabilă globală pentru factuality checker
factuality_checker = None

# Funcție pentru inițializarea checker-ului - o apelez la prima cerere
def init_factuality_checker():
    global factuality_checker
    if factuality_checker is None:
        try:
            from app.llm_module.factuality_checker import FactualityChecker
            logger.info("Inițializez FactualityChecker...")
            factuality_checker = FactualityChecker()
            logger.info("FactualityChecker inițializat cu succes!")
        except Exception as e:
            logger.error(f"Eroare la inițializarea FactualityChecker: {str(e)}")
            # Las factuality_checker None, aplicația va funcționa fără LLM

# Ruta principală - pagina de start
@app.route('/')
def index():
    """Pagina principală cu conținut recent"""
    # Găsesc ultimele chestii adăugate
    recent_content = Content.query.order_by(Content.created_at.desc()).limit(10).all()
    return render_template('index.html', contents=recent_content)

# NOUĂ RUTĂ: Analiză text manual
@app.route('/analyze-text', methods=['GET', 'POST'])
def analyze_manual_text():
    """Analizează text introdus manual - fără să îl salveze ca Content"""
    if request.method == 'POST':
        # Obțin datele din formular
        manual_text = request.form.get('manual_text', '').strip()
        title = request.form.get('title', 'Text manual').strip()
        
        if not manual_text:
            flash('Te rog să introduci textul pentru analiză.')
            return redirect(url_for('analyze_manual_text'))
        
        if len(manual_text) < 10:
            flash('Textul este prea scurt pentru analiză (minim 10 caractere).')
            return redirect(url_for('analyze_manual_text'))
        
        # Inițializez factuality checker dacă nu e gata
        if factuality_checker is None:
            init_factuality_checker()
        
        # Verific dacă modelul LLM e disponibil
        if not factuality_checker:
            flash('Analiza automată nu e disponibilă momentan. Verifică dependențele.')
            return redirect(url_for('analyze_manual_text'))
        
        try:
            logger.info(f"Analizez text manual: {len(manual_text)} caractere")
            
            # Analizez textul direct cu TinyLlama
            analysis_result = factuality_checker.analyze_text_content(manual_text, title)
            
            if analysis_result:
                logger.info("Analiză text manual completată cu succes")
                
                # Adaug informații suplimentare pentru template
                analysis_result['source_info'] = 'Text introdus manual'
                analysis_result['word_count'] = len(manual_text.split())
                analysis_result['char_count'] = len(manual_text)
                analysis_result['text_preview'] = manual_text[:200] + "..." if len(manual_text) > 200 else manual_text
                
                flash('Text analizat cu succes!', 'success')
                return render_template(
                    'analyze_manual.html',
                    analysis=analysis_result,
                    title=title,
                    text_preview=analysis_result['text_preview']
                )
            else:
                flash('Nu s-a putut realiza analiza textului.', 'error')
                
        except Exception as e:
            logger.error(f'Eroare la analiza textului manual: {str(e)}')
            flash(f'Eroare la analiza textului: {str(e)}', 'error')
    
    return render_template('analyze_manual.html')

# Înregistrarea utilizatorilor noi
@app.route('/register', methods=['GET', 'POST'])
def register():
    """Înregistrare utilizator nou"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Verifică dacă username-ul e deja luat
        if User.query.filter_by(username=username).first():
            flash('Acest username e deja folosit.')
            return redirect(url_for('register'))
        
        # Crează utilizator nou - TREBUIE HASH LA PAROLĂ!!! (TODO)
        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()
        
        flash('Te-ai înregistrat cu succes! Acum te poți autentifica.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

# Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    """Autentificare utilizator"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Caută utilizatorul
        user = User.query.filter_by(username=username).first()
        
        # Verifică dacă utilizatorul există și parola e corectă
        # TODO: Adaugă hashare parolă, e periculos așa!
        if user and user.password == password:
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Te-ai autentificat cu succes!')
            return redirect(url_for('index'))
        else:
            flash('Username sau parolă incorecte.')
    
    return render_template('login.html')

# Logout
@app.route('/logout')
def logout():
    """Delogare utilizator"""
    session.pop('user_id', None)
    session.pop('username', None)
    flash('Te-ai delogat cu succes.')
    return redirect(url_for('index'))

# Încărcare conținut - ACTUALIZAT pentru a suporta extragerea din URL și text manual
@app.route('/upload', methods=['GET', 'POST'])
def upload_content():
    """Încărcare conținut nou cu suport pentru URL-uri și text manual"""
    # Trebuie să fii logat
    if 'user_id' not in session:
        flash('Trebuie să fii autentificat pentru a încărca conținut.')
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        title = request.form.get('title')
        
        if not title:
            flash('Te rog să pui un titlu.')
            return redirect(url_for('upload_content'))
        
        # Verifică tipul de sursă
        source_type = request.form.get('source_type', 'file')
        
        # Dacă e text manual, redirecționez către analiza directă
        if source_type == 'manual_text':
            manual_text = request.form.get('manual_text', '').strip()
            if not manual_text:
                flash('Te rog să introduci textul pentru analiză.')
                return redirect(url_for('upload_content'))
            
            # Salvez temporar în sesiune și redirecționez
            session['temp_manual_text'] = manual_text
            session['temp_manual_title'] = title
            return redirect(url_for('analyze_manual_text'))
        
        # Creez obiectul de conținut pentru fișiere și URL-uri
        content = Content(
            title=title,
            uploaded_by_id=session['user_id']  # cine îl uploadează
        )
        
        # Verifică dacă e fișier sau link
        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            
            # Verifică tipul fișierului
            content_type = check_file_type(file.filename)
            if not content_type:
                flash('Acest tip de fișier nu e permis.')
                return redirect(url_for('upload_content'))
            
            # Salvează fișierul
            file_path = save_uploaded_file(file)
            
            # Setează informațiile despre conținut
            content.is_file = True
            content.content_type = content_type
            content.file_path = file_path
            
        elif request.form.get('url'):
            url = request.form.get('url')
            
            # Verifică dacă URL-ul e valid
            if not is_valid_url(url):
                flash('URL invalid. Te rog să introduci un URL valid (cu http:// sau https://).')
                return redirect(url_for('upload_content'))
            
            # Pentru URL-uri, încerc să extrag titlul automat dacă nu a fost specificat
            if title.strip() == "" or title.strip() == "Titlu implicit":
                try:
                    from app.utils import extract_text_from_url
                    url_result = extract_text_from_url(url)
                    if url_result['success'] and url_result['title'] and url_result['title'] != 'Fără titlu':
                        content.title = url_result['title']
                        logger.info(f"Titlu extras automat din URL: {url_result['title']}")
                except Exception as e:
                    logger.warning(f"Nu am putut extrage titlul din URL: {e}")
            
            # Setează informațiile despre conținut
            content.is_file = False
            content.content_type = request.form.get('content_type', 'text')
            content.url = url
            
        else:
            flash('Te rog să încarci un fișier, să pui un link sau să introduci text manual.')
            return redirect(url_for('upload_content'))
        
        # Salvez în baza de date
        db.session.add(content)
        db.session.commit()
        
        flash('Conținut încărcat cu succes!')
        return redirect(url_for('view_content', content_id=content.id))
    
    # Pentru GET request, verific dacă am text manual în sesiune
    manual_text = session.pop('temp_manual_text', '')
    manual_title = session.pop('temp_manual_title', '')
    
    return render_template('upload.html', manual_text=manual_text, manual_title=manual_title)

# Răsfoiește conținutul
@app.route('/browse')
def browse_content():
    """Răsfoiește conținutul cu filtrare opțională"""
    content_type = request.args.get('type')
    
    # Pornesc cu toate conținuturile
    query = Content.query
    
    # Aplic filtrul dacă e specificat
    if content_type in ['text', 'video', 'audio']:
        query = query.filter_by(content_type=content_type)
    
    # Sortez după data adăugării, cel mai recent primul
    all_content = query.order_by(Content.created_at.desc()).all()
    
    return render_template('browse.html', contents=all_content)

# Vizualizare conținut
@app.route('/content/<int:content_id>')
def view_content(content_id):
    """Vizualizează un conținut specific și evaluările sale"""
    content = Content.query.get_or_404(content_id)
    
    # Calculez ratingurile medii
    avg_ratings = content.get_avg_ratings()
    
    # Iau toate ratingurile pentru acest conținut
    ratings = Rating.query.filter_by(content_id=content.id).all()
    
    # Văd dacă avem analiză AI salvată
    ai_analysis = content.get_ai_analysis()
    
    return render_template(
        'content.html', 
        content=content,
        average_ratings=avg_ratings,
        ratings=ratings,
        ai_analysis=ai_analysis
    )

# Evaluare conținut
@app.route('/content/<int:content_id>/rate', methods=['GET', 'POST'])
def rate_content(content_id):
    """Evaluează un conținut specific"""
    # Trebuie să fii logat
    if 'user_id' not in session:
        flash('Trebuie să fii autentificat pentru a evalua conținut.')
        return redirect(url_for('login'))
    
    # Găsesc conținutul
    content = Content.query.get_or_404(content_id)
    
    if request.method == 'POST':
        # Iau valorile din formular
        coherence = int(request.form.get('coherence', 5))
        truth = int(request.form.get('truth', 5))
        attractiveness = int(request.form.get('attractiveness', 5))
        comment = request.form.get('comment', '')
        
        # Verific dacă valorile sunt între 1 și 10
        coherence = max(1, min(10, coherence))
        truth = max(1, min(10, truth))
        attractiveness = max(1, min(10, attractiveness))
        
        # Verific dacă utilizatorul a evaluat deja acest conținut
        existing_rating = Rating.query.filter_by(
            content_id=content.id,
            user_id=session['user_id']
        ).first()
        
        if existing_rating:
            # Actualizez evaluarea existentă
            existing_rating.coherence = coherence
            existing_rating.truth = truth
            existing_rating.attractiveness = attractiveness
            existing_rating.comment = comment
            flash('Evaluarea ta a fost actualizată!')
        else:
            # Creez o evaluare nouă
            rating = Rating(
                content_id=content.id,
                user_id=session['user_id'],
                coherence=coherence,
                truth=truth,
                attractiveness=attractiveness,
                comment=comment
            )
            db.session.add(rating)
            flash('Evaluarea ta a fost înregistrată!')
        
        db.session.commit()
        return redirect(url_for('view_content', content_id=content.id))
    
    # Verific dacă utilizatorul a evaluat deja acest conținut
    existing_rating = None
    if 'user_id' in session:
        existing_rating = Rating.query.filter_by(
            content_id=content.id,
            user_id=session['user_id']
        ).first()
    
    return render_template('rate.html', content=content, existing_rating=existing_rating)

# Analiză factualitate cu LLM - ACTUALIZAT pentru suport URL
@app.route('/content/<int:content_id>/analyze', methods=['GET'])
def analyze_content(content_id):
    """Analiză automată a conținutului cu LLM - acum suportă și URL-uri"""
    # Trebuie să fii logat
    if 'user_id' not in session:
        flash('Trebuie să fii autentificat pentru a analiza conținutul.')
        return redirect(url_for('login'))
    
    # Găsesc conținutul
    content = Content.query.get_or_404(content_id)
    
    # Verific dacă avem deja o analiză salvată
    existing_analysis = content.get_ai_analysis()
    if existing_analysis:
        return render_template(
            'analyze.html', 
            content=content,
            analysis=existing_analysis
        )
    
    # Inițializez factuality checker dacă nu e gata
    if factuality_checker is None:
        init_factuality_checker()
    
    # Verific dacă modelul LLM e disponibil
    if not factuality_checker:
        flash('Analiza automată nu e disponibilă momentan. Probabil ai nevoie să instalezi torch și celelalte dependențe.')
        return redirect(url_for('view_content', content_id=content_id))
    
    # Obțin conținutul pentru analiză (funcționează atât pentru fișiere cât și URL-uri)
    content_result = get_content_for_analysis(content)
    
    if not content_result['success']:
        flash(f'Nu s-a putut obține conținutul pentru analiză: {content_result["error"]}')
        return redirect(url_for('view_content', content_id=content_id))
    
    # Analizez conținutul cu TinyLlama
    try:
        logger.info(f"Analizez conținutul {content.id} cu TinyLlama...")
        analysis_result = factuality_checker.analyze_text_content(
            content_result['text'], 
            content.title
        )
        
        if analysis_result:
            # Salvez rezultatul analizei
            store_ai_analysis(content, analysis_result)
            db.session.commit()
            logger.info(f"Analiză completată pentru conținutul {content.id}")
            
            # Adaug informații despre sursa analizată
            if content_result['source'] == 'url':
                flash(f'Text analizat cu succes din URL: {content.url}', 'success')
            elif content_result['source'] == 'file':
                flash(f'Fișier analizat cu succes: {content.file_path.split("/")[-1] if content.file_path else "necunoscut"}', 'success')
        else:
            flash('Nu s-a putut realiza analiza conținutului.', 'error')
            analysis_result = None
            
    except Exception as e:
        logger.error(f'Eroare la analiza conținutului {content.id}: {str(e)}')
        flash(f'Eroare la analiza conținutului: {str(e)}', 'error')
        analysis_result = None
    
    return render_template(
        'analyze.html',
        content=content,
        analysis=analysis_result
    )

# API pentru analiza de factualitate
@app.route('/api/analyze_factuality', methods=['POST'])
def analyze_factuality():
    """API endpoint pentru analiza factualității unui text"""
    # Inițializez factuality checker dacă nu e gata
    if factuality_checker is None:
        init_factuality_checker()
    
    # Verific dacă modelul LLM e disponibil
    if not factuality_checker:
        return jsonify({
            "error": "Modelul LLM nu e disponibil. Verifică instalarea dependențelor."
        }), 500
    
    # Verific datele de intrare
    data = request.json
    if not data or 'text' not in data:
        return jsonify({
            "error": "Trebuie să furnizezi textul pentru analiză în format JSON: {'text': 'textul de analizat'}"
        }), 400
    
    text = data['text']
    title = data.get('title', '')
    
    # Analizez textul
    analysis = factuality_checker.analyze_text_content(text, title)
    
    return jsonify(analysis)

# NOUĂ RUTĂ: API pentru extragerea textului din URL
@app.route('/api/extract-url', methods=['POST'])
def api_extract_url():
    """API endpoint pentru extragerea textului din URL-uri"""
    try:
        data = request.get_json()
        url = data.get('url') if data else None
        
        if not url:
            return jsonify({
                'success': False, 
                'error': 'URL este necesar. Trimite JSON cu {"url": "https://example.com"}'
            }), 400
        
        # Folosesc funcția din utils
        from app.utils import extract_text_from_url
        result = extract_text_from_url(url)
        
        if result['success']:
            return jsonify({
                'success': True,
                'title': result['title'],
                'text': result['text'][:2000] + "..." if len(result['text']) > 2000 else result['text'],  # Limitez pentru API
                'word_count': result.get('word_count', 0),
                'char_count': result.get('char_count', 0),
                'url': url
            })
        else:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
            
    except Exception as e:
        logger.error(f"Eroare în api_extract_url: {e}")
        return jsonify({
            'success': False, 
            'error': f'Eroare la procesarea cererii: {str(e)}'
        }), 500

# NOUĂ RUTĂ: Testare extragere URL
@app.route('/test-url-extraction', methods=['GET', 'POST'])
def test_url_extraction():
    """Pagină pentru testarea extragerii de text din URL-uri"""
    if request.method == 'POST':
        url = request.form.get('url')
        
        if not url:
            flash('Te rog să introduci un URL.')
            return render_template('test_url.html')
        
        # Testez extragerea
        from app.utils import extract_text_from_url
        result = extract_text_from_url(url)
        
        return render_template('test_url.html', result=result, test_url=url)
    
    return render_template('test_url.html')

# Rută pentru testarea modelului LLM
@app.route('/test-llm')
def test_llm():
    """Pagină de test pentru modelul LLM"""
    # Doar pentru admini, dar nu avem implementat roluri încă
    if 'user_id' not in session:
        flash('Trebuie să fii autentificat pentru a testa LLM-ul.')
        return redirect(url_for('login'))
    
    # Inițializez factuality checker dacă nu e gata
    if factuality_checker is None:
        init_factuality_checker()
    
    try:
        # Dacă modelul e disponibil, îl testez cu un text simplu
        if factuality_checker:
            test_result = factuality_checker.analyze_text_content(
                "Pământul este a treia planetă de la Soare și singura planetă din sistemul solar care susține viață."
            )
            
            return render_template(
                'test_llm.html', 
                result=test_result,
                success=True
            )
        else:
            raise Exception("Modelul LLM nu e disponibil. Verifică dependențele.")
    except Exception as e:
        return render_template(
            'test_llm.html',
            error=str(e),
            success=False
        )