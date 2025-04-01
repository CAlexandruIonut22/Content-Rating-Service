# app/models.py
from app import db
from datetime import datetime

class User(db.Model):
    """Model utilizator pentru autentificare"""
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)  # FIXME: Trebuie hash-uite parolele!
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Content(db.Model):
    """Model pentru conținutul evaluat (fișiere sau linkuri)"""
    id = db.Column(db.Integer, primary_key=True)
    itle = db.Column(db.String(200), nullable=False)
    type = db.Column(db.String(20), nullable=False)  # 'text', 'video', 'audio'
    is_file = db.Column(db.Boolean, default=False)
    
    # Path pentru fisiere incarcate
    file_path = db.Column(db.String(255), nullable=True)
    
    # URL pentru linkuri
    url = db.Column(db.String(500), nullable=True)
    
    # Data la creare
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Analiza AI
    ai_factuality_score = db.Column(db.Float, nullable=True)
    ai_analysis_data = db.Column(db.Text, nullable=True)  # JSON cu rezultatele complete
    
    # Relație cu evaluările
    ratings = db.relationship('Rating', backref='content', lazy=True)
    
    def get_avg_ratings(self):
        """Calculează ratingurile medii pentru acest conținut"""
        ratings = self.ratings
        if not ratings:
            return {'coherence': 0, 'truth': 0, 'attractiveness': 0, 'overall': 0}
        
        # Calculează totalurile pentru fiecare criteriu
        c_total = sum(r.coherence for r in ratings)
        t_total = sum(r.truth for r in ratings)
        a_total = sum(r.attractiveness for r in ratings)
        
        count = len(ratings)
        c_avg = c_total / count
        t_avg = t_total / count
        a_avg = a_total / count
        
        # Calculează scorul general (media celor trei criterii)
        overall = (c_avg + t_avg + a_avg) / 3
        
        # Rotunjește la o zecimala
        return {
            'coherence': round(c_avg, 1),
            'truth': round(t_avg, 1),
            'attractiveness': round(a_avg, 1),
            'overall': round(overall, 1)
        }

class Rating(db.Model):
    """Model pentru evaluările individuale ale conținutului"""
    id = db.Column(db.Integer, primary_key=True)
    content_id = db.Column(db.Integer, db.ForeignKey('content.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    # Cele trei criterii de evaluare (scară 1-10)
    coherence = db.Column(db.Integer, nullable=False)  # cât de coerent este
    truth = db.Column(db.Integer, nullable=False)      # cât de adevărat este
    attractiveness = db.Column(db.Integer, nullable=False)  # cât de bine prezentat este 
    
    # Comentariu opțional rating
    comment = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relație cu utilizatorul
    user = db.relationship('User', backref='ratings')