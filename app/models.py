from app import db 
from datetime import datetime
import json

class User(db.Model):
    """Modelul pentru utilizatori - simplu deocamdată, fără email verification etc."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)  # TODO: Trebuie hash-uit!! Risc de securitate
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relații - poate nu folosesc acum dar vor fi utile mai târziu
    uploads = db.relationship('Content', backref='uploaded_by', lazy=True, 
                              foreign_keys='Content.uploaded_by_id')
    ratings = db.relationship('Rating', backref='user', lazy=True)
    
    def __repr__(self):
        return f'<User {self.username}>'

class Content(db.Model):
    """Model pentru conținutul ce va fi evaluat - fișiere sau linkuri"""
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content_type = db.Column(db.String(20), nullable=False)  # 'text', 'video', 'audio'
    is_file = db.Column(db.Boolean, default=False)
    
    # Pentru fișiere
    file_path = db.Column(db.String(255), nullable=True)
    
    # Pentru linkuri
    url = db.Column(db.String(500), nullable=True)
    
    # Cine a uploadat
    uploaded_by_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    # Când a fost adăugat
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Aici stocăm rezultatul analizei LLM
    ai_factuality_score = db.Column(db.Float, nullable=True)  # scorul de 1-10
    ai_analysis_data = db.Column(db.Text, nullable=True)  # JSON cu toate detaliile
    
    # Relația cu ratingurile
    ratings = db.relationship('Rating', backref='content', lazy=True)
    
    def get_avg_ratings(self):
        """Calculează media ratingurilor - probabil e un mod mai eficient, dar merge și așa"""
        ratings = self.ratings
        if not ratings:
            return {'coherence': 0, 'truth': 0, 'attractiveness': 0, 'overall': 0}
        
        c_total = 0
        t_total = 0
        a_total = 0
        
        for r in ratings:
            c_total += r.coherence
            t_total += r.truth
            a_total += r.attractiveness
            
        count = len(ratings)
        c_avg = c_total / count
        t_avg = t_total / count
        a_avg = a_total / count
        
        # Media celor 3 criterii = overall
        overall = (c_avg + t_avg + a_avg) / 3
        
        # Rotunjesc că nu-mi plac zecimalele lungi
        return {
            'coherence': round(c_avg, 1),
            'truth': round(t_avg, 1),
            'attractiveness': round(a_avg, 1),
            'overall': round(overall, 1)
        }
    
    def get_ai_analysis(self):
        """Returnează datele de analiză AI dacă există"""
        if not self.ai_analysis_data:
            return None
        
        try:
            return json.loads(self.ai_analysis_data)
        except:
            return None  # json invalid sau ceva
    
    def __repr__(self):
        return f'<Content {self.title}>'

class Rating(db.Model):
    """Evaluările date de utilizatori"""
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    # Cele 3 criterii de rating
    coherence = db.Column(db.Integer, nullable=False)  # cât de coerent e
    truth = db.Column(db.Integer, nullable=False)      # cât de adevărat e
    attractiveness = db.Column(db.Integer, nullable=False)  # cât de frumos prezentat e
    
    # Un comentariu opțional
    comment = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Rating {self.id} for Content {self.content_id}>'