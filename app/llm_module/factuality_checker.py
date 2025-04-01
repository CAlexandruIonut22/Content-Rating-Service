# app/llm_module/factuality_checker.py
from .model_handler import ModelHandler
import logging
import json
import re

logger = logging.getLogger(__name__)

class FactualityChecker:
    def __init__(self):
        self.model_handler = ModelHandler()
        if not self.model_handler.initialized:
            self.model_handler.initialize()
    
    def analyze_text_content(self, text, title=None):
        """Analizează conținutul textual și returnează evaluarea factualității."""
        if not text or len(text.strip()) < 10:
            return {
                "factuality_score": 0,
                "confidence": 0,
                "reasoning": "Text prea scurt pentru analiză.",
                "questionable_claims": []
            }
        
        # Limitează textul pentru a evita token overflow
        max_text_length = 1500
        if len(text) > max_text_length:
            analyzed_text = text[:max_text_length] + "..."
        else:
            analyzed_text = text
        
        prompt = self._create_factuality_prompt(analyzed_text, title)
        
        try:
            response = self.model_handler.generate_response(prompt, max_length=1024, temperature=0.3)
            
            # Parsează răspunsul pentru a extrage evaluarea structurată
            parsed_result = self._parse_factuality_response(response)
            return parsed_result
        except Exception as e:
            logger.error(f"Eroare la analiza factualității: {str(e)}")
            return {
                "factuality_score": 0,
                "confidence": 0,
                "reasoning": f"Eroare la procesare: {str(e)}",
                "questionable_claims": []
            }
    
    def _create_factuality_prompt(self, text, title=None):
        """Creează un prompt pentru analiza factualității."""
        title_text = f"Titlu: {title}\n\n" if title else ""
        
        prompt = f"""Analizează următorul text din perspectiva factualității și acurateței informațiilor. 
        {title_text}
        Text: "{text}"
        
        Evaluează textul de mai sus și oferă:
        1. Un scor de factualitate între 1 și 10 (unde 1=complet fals, 10=complet adevărat)
        2. Nivelul tău de încredere în evaluare (între 1 și 10)
        3. Raționamentul care justifică evaluarea
        4. O listă cu afirmațiile care par problematice sau necesită verificare

        Răspunde în format JSON cu următoarea structură exactă:
        {{
            "factuality_score": (scor între 1-10),
            "confidence": (nivel de încredere între 1-10),
            "reasoning": "Explicația ta detaliată",
            "questionable_claims": ["afirmație 1", "afirmație 2", ...]
        }}
        
        Returnează doar JSON-ul, fără alt text.
        """
        
        return prompt
    
    def _parse_factuality_response(self, response):
        """Parsează răspunsul LLM pentru a extrage informațiile structurate."""
        try:
            # Încearcă să găsească orice structură JSON în răspuns
            json_match = re.search(r'({[\s\S]*})', response)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                
                # Validează structura rezultatului
                required_keys = ["factuality_score", "confidence", "reasoning", "questionable_claims"]
                for key in required_keys:
                    if key not in result:
                        result[key] = 0 if key in ["factuality_score", "confidence"] else ([] if key == "questionable_claims" else "Informație lipsă")
                
                # Asigură-te că scorurile sunt numere între 1 și 10
                result["factuality_score"] = min(max(int(result["factuality_score"]), 1), 10)
                result["confidence"] = min(max(int(result["confidence"]), 1), 10)
                
                return result
            else:
                # Fallback dacă nu găsim un JSON valid
                return {
                    "factuality_score": 5,
                    "confidence": 5,
                    "reasoning": "Nu s-a putut extrage o analiză structurată. Răspuns brut: " + response[:200] + "...",
                    "questionable_claims": []
                }
        except Exception as e:
            logger.error(f"Eroare la parsarea răspunsului: {str(e)}")
            logger.error(f"Răspuns original: {response}")
            return {
                "factuality_score": 0,
                "confidence": 0,
                "reasoning": f"Eroare la procesarea răspunsului: {str(e)}",
                "questionable_claims": []
            }