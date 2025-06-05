# -*- coding: utf-8 -*-
# app/url_extractor.py
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class URLTextExtractor:
    """Extrage text din URL-uri pentru analiza cu LLM"""
    
    def __init__(self):
        """Initializeaza extractorul cu configurari optimizate"""
        self.session = requests.Session()
        
        # Configurez retry strategy pentru conexiuni instabile
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Headers ca sa nu fim blocati de site-uri
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ro-RO,ro;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }

    def validate_url(self, url):
        """Verifica daca URL-ul e valid"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def extract_text_from_url(self, url, timeout=15):
        """
        Extrage textul principal dintr-un URL
        
        Args:
            url (str): URL-ul de la care sa extraga textul
            timeout (int): Timeout pentru request in secunde
            
        Returns:
            dict: Rezultatul extragerii cu textul si metadata
        """
        logger.info(f"Incerc sa extrag text din: {url}")
        
        if not self.validate_url(url):
            logger.warning(f"URL invalid: {url}")
            return {
                'success': False,
                'error': 'URL invalid - verifica formatul (trebuie sa inceapa cu http:// sau https://)',
                'text': '',
                'title': '',
                'url': url
            }

        try:
            # Trimit request-ul
            logger.info(f"Trimit request catre {url}...")
            response = self.session.get(
                url, 
                headers=self.headers, 
                timeout=timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Verific content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                logger.warning(f"Content type nesuportat: {content_type}")
                return {
                    'success': False,
                    'error': f'Tipul de continut nu este HTML: {content_type}. Pot procesa doar pagini web.',
                    'text': '',
                    'title': '',
                    'url': url
                }

            # Parse HTML cu BeautifulSoup
            logger.info("Parsez HTML-ul...")
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extrag titlul
            title = self._extract_title(soup)
            logger.info(f"Titlu extras: {title}")
            
            # Extrag textul principal
            main_text = self._extract_main_text(soup)
            
            # Curat textul
            cleaned_text = self._clean_text(main_text)
            
            if not cleaned_text.strip():
                logger.warning("Nu s-a putut extrage text din pagina")
                return {
                    'success': False,
                    'error': 'Pagina nu contine text care sa poata fi extras automat',
                    'text': '',
                    'title': title,
                    'url': url
                }
            
            logger.info(f"Text extras cu succes: {len(cleaned_text)} caractere")
            return {
                'success': True,
                'error': None,
                'text': cleaned_text,
                'title': title,
                'url': url,
                'word_count': len(cleaned_text.split()),
                'char_count': len(cleaned_text)
            }
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout la request pentru {url}")
            return {
                'success': False,
                'error': f'Site-ul nu raspunde in timp util (timeout dupa {timeout} secunde)',
                'text': '',
                'title': '',
                'url': url
            }
        except requests.exceptions.ConnectionError:
            logger.error(f"Eroare de conexiune pentru {url}")
            return {
                'success': False,
                'error': 'Nu ma pot conecta la acest site. Verifica conexiunea la internet.',
                'text': '',
                'title': '',
                'url': url
            }
        except requests.exceptions.HTTPError as e:
            logger.error(f"Eroare HTTP pentru {url}: {e}")
            return {
                'success': False,
                'error': f'Site-ul a returnat o eroare: {e} (probabil pagina nu exista)',
                'text': '',
                'title': '',
                'url': url
            }
        except Exception as e:
            logger.error(f"Eroare neasteptata pentru {url}: {e}")
            return {
                'success': False,
                'error': f'Eroare la procesarea paginii: {str(e)}',
                'text': '',
                'title': '',
                'url': url
            }

    def _extract_title(self, soup):
        """Extrage titlul paginii incercand mai multe metode"""
        # Incerc mai multe selectoare pentru titlu
        title_selectors = [
            'title',                    # Titlul standard HTML
            'h1',                       # Primul heading
            '[property="og:title"]',    # Open Graph title
            '[name="twitter:title"]',   # Twitter card title
            '.title',                   # Clasa .title
            '.headline',                # Pentru site-uri de stiri
            '.post-title',              # Pentru bloguri
            '.entry-title'              # Pentru WordPress
        ]
        
        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                if hasattr(element, 'get_text'):
                    title = element.get_text(strip=True)
                else:
                    title = element.get('content', '')
                
                if title and len(title.strip()) > 0:
                    return title.strip()[:200]  # Limitez la 200 caractere
        
        return 'Fara titlu'

    def _extract_main_text(self, soup):
        """Extrage textul principal din pagina eliminand elementele nedorite"""
        # Elimin elementele care nu contin continut util
        for unwanted in soup(["script", "style", "nav", "header", "footer", 
                              "aside", "form", "button", "input", "select", 
                              "textarea", "iframe", "noscript", "meta", "link"]):
            unwanted.decompose()
        
        # Incerc sa gasesc zona de continut principal
        main_content_selectors = [
            'article',           # HTML5 semantic
            '[role="main"]',     # ARIA role
            'main',              # HTML5 main element
            '.content',          # Clase comune
            '.post-content',
            '.entry-content',
            '.article-content',
            '.main-content',
            '#content',          # ID-uri comune
            '#main',
            '#post',
            '.post-body',
            '.story-body'        # Pentru site-uri de stiri
        ]
        
        # Caut zona principala de continut
        main_element = None
        for selector in main_content_selectors:
            element = soup.select_one(selector)
            if element:
                main_element = element
                logger.info(f"Gasit continut principal cu selectorul: {selector}")
                break
        
        # Daca nu gasesc zona principala, folosesc body-ul
        if not main_element:
            main_element = soup.find('body') or soup
            logger.info("Folosesc body-ul ca zona de continut")
        
        # Extrag paragrafele si headingurile
        text_elements = main_element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote'])
        
        texts = []
        for element in text_elements:
            text = element.get_text(strip=True)
            # Filtrez textele care sunt prea scurte sau par a fi navigare/metadata
            if text and len(text) > 15 and not self._is_navigation_text(text):
                texts.append(text)
        
        return '\n\n'.join(texts)

    def _is_navigation_text(self, text):
        """Verifica daca textul pare sa fie navigare sau metadata in loc de continut"""
        # Lista de cuvinte care sugereaza navigare/metadata
        nav_indicators = [
            'meniu', 'navigare', 'login', 'sign in', 'register', 'home', 'contact',
            'despre noi', 'politica', 'cookie', 'terms', 'privacy', 'copyright',
            'follow us', 'share', 'like', 'tweet', 'facebook', 'instagram',
            'next', 'previous', 'page', 'comments', 'reply', 'edit', 'delete'
        ]
        
        text_lower = text.lower()
        
        # Daca textul e prea scurt si contine indicatori de navigare
        if len(text) < 50:
            for indicator in nav_indicators:
                if indicator in text_lower:
                    return True
        
        return False

    def _clean_text(self, text):
        """Curata si formateaza textul extras"""
        if not text:
            return ''
        
        # Elimin spatiile multiple si newline-urile excesive
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Elimin caractere speciale problematice dar pastrez punctuatia si diacriticele
        text = re.sub(r'[^\w\s\.,!?;:()\-"\'ăâîșțĂÂÎȘȚáéíóúÁÉÍÓÚ]', ' ', text)
        
        # Elimin liniile foarte scurte care par a fi metadata
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if len(line) > 10:  # Pastrez doar liniile cu mai mult de 10 caractere
                cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # Elimin spatiile de la inceput si sfarsit
        return text.strip()


# Instanta globala pentru a fi folosita in aplicatie
url_extractor = URLTextExtractor()