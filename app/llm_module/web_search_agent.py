# app/llm_module/web_search_agent.py - Optimizat pentru TinyLlama
import sys
import os
import requests
import json
import logging
import re
from typing import Dict, List, Optional

# Fix pentru import-uri
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from .model_handler import ModelHandler

logger = logging.getLogger(__name__)

class WebSearchAgent:
    """
    Agent web search optimizat pentru TinyLlama
    Prompturi simple și verificări rapide
    """
    
    def __init__(self, search_api_key: Optional[str] = None):
        self.search_api_key = search_api_key
        self.search_enabled = search_api_key is not None
        self.model_handler = None
        
        # Inițializează model handler doar dacă e necesar
        self._init_model_handler()
        
        logger.info(f"WebSearchAgent inițializat (search {'activat' if self.search_enabled else 'dezactivat'})")
    
    def _init_model_handler(self):
        """Inițializează model handler pentru analiză"""
        try:
            self.model_handler = ModelHandler()
            if not self.model_handler.initialized:
                # Folosește TinyLlama pentru analiză web
                self.model_handler.initialize(
                    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    use_4bit=False
                )
            logger.info("Model handler pentru web search inițializat")
        except Exception as e:
            logger.warning(f"Nu pot inițializa model handler pentru web search: {e}")
            self.model_handler = None
    
    def search_web(self, query: str, max_results: int = 3) -> List[Dict]:
        """Caută pe web folosind Tavily API"""
        if not self.search_enabled:
            logger.warning("Web search dezactivat - lipsește API key")
            return []
        
        try:
            logger.info(f"🔍 Caut pe web: '{query}'")
            
            # Tavily Search API
            url = "https://api.tavily.com/search"
            
            payload = {
                "api_key": self.search_api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "include_raw_content": False,
                "max_results": max_results,
                "include_domains": [
                    "wikipedia.org", "britannica.com", "reuters.com", 
                    "bbc.com", "cnn.com", "mediafax.ro", "digi24.ro"
                ],
                "exclude_domains": [
                    "reddit.com", "quora.com", "yahoo.answers.com"
                ]
            }
            
            response = requests.post(url, json=payload, timeout=15)
            response.raise_for_status()
            
            results = response.json()
            search_results = []
            
            for result in results.get('results', []):
                search_results.append({
                    'title': result.get('title', ''),
                    'content': result.get('content', ''),
                    'url': result.get('url', ''),
                    'score': result.get('score', 0)
                })
            
            logger.info(f"✅ Găsite {len(search_results)} rezultate pentru: {query}")
            return search_results
            
        except Exception as e:
            logger.error(f"❌ Eroare căutare web: {e}")
            return []
    
    def extract_claims_for_verification(self, text: str) -> List[str]:
        """
        Extrage afirmații din text pentru verificare
        Versiune simplificată pentru TinyLlama
        """
        if not self.model_handler:
            # Fallback simplu fără LLM
            return self._extract_claims_simple(text)
        
        # Prompt foarte simplu pentru TinyLlama
        prompt = f"""Citește textul și găsește 3 lucruri importante care pot fi verificate:

Text: {text[:800]}

Scrie doar 3 afirmații importante, fiecare pe o linie nouă.
Nu pune numere sau puncte.
Exemplu:
România are 19 milioane locuitori
Bucureștiul este capitala României
UE a fost creată în 1993"""
        
        try:
            response = self.model_handler.generate_response(
                prompt,
                max_new_tokens=150,
                temperature=0.5,
                do_sample=True
            )
            
            # Extrage afirmațiile din răspuns
            claims = []
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                # Curăță linia de prefixuri comune
                line = re.sub(r'^[-*•\d+\.)]\s*', '', line)
                
                if len(line) > 15 and len(line) < 200:  # Lungime rezonabilă
                    claims.append(line)
            
            # Limitează la 3 afirmații maxim
            return claims[:3]
            
        except Exception as e:
            logger.error(f"Eroare extragere afirmații cu TinyLlama: {e}")
            return self._extract_claims_simple(text)
    
    def _extract_claims_simple(self, text):
        """Extragere simplă de afirmații fără LLM"""
        # Împarte în propoziții
        sentences = re.split(r'[.!?]', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            
            # Filtrează propozițiile care par să facă afirmații factuale
            if (len(sentence) > 20 and 
                len(sentence) < 150 and
                any(word in sentence.lower() for word in 
                    ['este', 'sunt', 'a fost', 'au fost', 'are', 'au', 'în', 'cu', 'de', 'la'])):
                claims.append(sentence)
        
        return claims[:3]  # Maxim 3
    
    def verify_claim_with_search(self, claim: str) -> Dict:
        """Verifică o afirmație prin căutare web + TinyLlama"""
        # Creează query simplu pentru căutare
        search_query = self._simplify_claim_for_search(claim)
        
        # Caută pe web
        search_results = self.search_web(search_query, max_results=2)
        
        if not search_results:
            return {
                'claim': claim,
                'verification_status': 'no_sources',
                'confidence': 0,
                'explanation': 'Nu s-au găsit surse pentru verificare',
                'sources_used': []
            }
        
        # Pregătește context pentru TinyLlama
        context = self._prepare_context_for_tinyllama(search_results)
        
        # Verifică cu TinyLlama
        return self._verify_with_tinyllama(claim, context, search_results)
    
    def _simplify_claim_for_search(self, claim):
        """Simplifică afirmația pentru căutare web"""
        # Elimină cuvinte comune care nu ajută la căutare
        stop_words = ['este', 'sunt', 'a fost', 'au fost', 'că', 'de', 'la', 'în', 'cu']
        
        words = claim.split()
        filtered_words = [word for word in words if word.lower() not in stop_words]
        
        # Păstrează doar primele 6 cuvinte importante
        search_query = ' '.join(filtered_words[:6])
        
        return search_query
    
    def _prepare_context_for_tinyllama(self, search_results):
        """Pregătește contextul pentru TinyLlama (foarte scurt)"""
        context = ""
        
        for i, result in enumerate(search_results[:2], 1):  # Maxim 2 surse
            title = result.get('title', '')[:100]
            content = result.get('content', '')[:200]  # Foarte scurt pentru TinyLlama
            
            context += f"Sursa {i}: {title}\n{content}\n\n"
        
        return context[:600]  # Limitează contextul total
    
    def _verify_with_tinyllama(self, claim, context, search_results):
        """Verifică afirmația cu TinyLlama folosind contextul web"""
        if not self.model_handler:
            return self._manual_verification(claim, context, search_results)
        
        # Prompt foarte simplu pentru TinyLlama
        prompt = f"""Afirmația: {claim}

Ce spun sursele de pe internet:
{context}

Este afirmația adevărată sau falsă?
Răspunde doar:
ADEVĂRAT - dacă sursele confirmă
FALS - dacă sursele contrazic  
NECLAR - dacă nu e destulă informație

Apoi explică pe scurt de ce."""
        
        try:
            response = self.model_handler.generate_response(
                prompt,
                max_new_tokens=100,
                temperature=0.3,
                do_sample=True
            )
            
            # Parsează răspunsul TinyLlama
            verification_result = self._parse_verification_response(response, claim, search_results)
            return verification_result
            
        except Exception as e:
            logger.error(f"Eroare verificare cu TinyLlama: {e}")
            return self._manual_verification(claim, context, search_results)
    
    def _parse_verification_response(self, response, claim, search_results):
        """Parsează răspunsul de verificare de la TinyLlama"""
        response_lower = response.lower()
        
        # Detectează statusul
        if 'adevărat' in response_lower or 'confirm' in response_lower:
            status = 'adevarata'
            confidence = 7
        elif 'fals' in response_lower or 'contrazic' in response_lower:
            status = 'falsa'
            confidence = 7
        elif 'neclar' in response_lower or 'insuficient' in response_lower:
            status = 'neconcludenta'
            confidence = 4
        else:
            # Fallback bazat pe cuvinte cheie
            positive_words = ['da', 'corect', 'exact', 'confirmat']
            negative_words = ['nu', 'greșit', 'incorect', 'fals']
            
            pos_count = sum(1 for word in positive_words if word in response_lower)
            neg_count = sum(1 for word in negative_words if word in response_lower)
            
            if neg_count > pos_count:
                status = 'falsa'
                confidence = 5
            elif pos_count > neg_count:
                status = 'adevarata'
                confidence = 5
            else:
                status = 'neconcludenta'
                confidence = 3
        
        # Extrage explicația
        explanation = response[:150] + "..." if len(response) > 150 else response
        
        return {
            'claim': claim,
            'verification_status': status,
            'confidence': confidence,
            'explanation': explanation,
            'sources_used': [r['url'] for r in search_results[:2]]
        }
    
    def _manual_verification(self, claim, context, search_results):
        """Verificare manuală fără LLM"""
        # Analiză simplă bazată pe cuvinte cheie
        claim_lower = claim.lower()
        context_lower = context.lower()
        
        # Extrage cuvinte cheie din afirmație
        claim_words = set(re.findall(r'\b\w+\b', claim_lower))
        context_words = set(re.findall(r'\b\w+\b', context_lower))
        
        # Calculează overlap
        overlap = len(claim_words.intersection(context_words))
        total_words = len(claim_words)
        
        if total_words == 0:
            confidence = 0
            status = 'neconcludenta'
        else:
            overlap_ratio = overlap / total_words
            
            if overlap_ratio > 0.6:
                status = 'adevarata'
                confidence = 6
            elif overlap_ratio > 0.3:
                status = 'partial_adevarata'
                confidence = 4
            else:
                status = 'neconcludenta'
                confidence = 3
        
        return {
            'claim': claim,
            'verification_status': status,
            'confidence': confidence,
            'explanation': f'Verificare automată: overlap {overlap}/{total_words} cuvinte cu sursele găsite',
            'sources_used': [r['url'] for r in search_results[:2]]
        }