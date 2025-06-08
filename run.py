# -*- coding: utf-8 -*-
import sys
import os

# Configurare encoding pentru Windows
if sys.platform.startswith('win'):
    # Setez encoding-ul pentru stdout/stderr
    import codecs
    if hasattr(sys.stdout, 'detach'):
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    if hasattr(sys.stderr, 'detach'):
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    
    # Setez variabila de mediu pentru encoding
    os.environ['PYTHONIOENCODING'] = 'utf-8'

from app import app
import logging

if __name__ == '__main__':
    print("""
    ------------------------------------------------
    Content Rating Service - Proiect Licenta
    ------------------------------------------------
    Aplicatia se porneste acum pe http://127.0.0.1:5000
    Apasa CTRL+C pentru a opri
    ------------------------------------------------
    """)
    # Logez si in consola ca sa vad ce se intampla
    logging.basicConfig(level=logging.INFO)
    #app.run(debug=True, host='127.0.0.1', port=5000)
