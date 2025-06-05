# app/llm_module/factuality_checker.py - VERSIUNEA CORECTATĂ

from .model_handler import ModelHandler
from app.config import Config
import logging
import json
import re

logger = logging.getLogger(__name__)

class FactualityChecker:
    """Clasa pentru verificarea factualității textului folosind modelul LLM UPGRADED"""
    
    def __init__(self):
        """Inițializează checker-ul cu configurația corectă"""
        self.model_handler = ModelHandler()
        if not self.model_handler.initialized:
            logger.info(f"Initializez modelul {Config.LLM_MODEL_ID}...")
            try:
                # FOLOSEȘTE doar parametrii care există în model_handler.py ACTUAL
                self.model_handler.initialize(
                    model_id=Config.LLM_MODEL_ID,
                    cache_dir=getattr(Config, 'LLM_CACHE_DIR', './model_cache'),
                    use_4bit=getattr(Config, 'LLM_USE_4BIT', True)
                    # Eliminat force_gpu și hf_token până actualizezi model_handler.py
                )
                logger.info(f"✅ Model {Config.LLM_MODEL_ID} inițializat cu succes!")
            except Exception as e:
                logger.error(f"❌ Eroare la inițializarea modelului: {e}")
                # Fallback la TinyLlama dacă modelul principal eșuează
                logger.info("🔄 Încerc fallback la TinyLlama...")
                try:
                    self.model_handler.initialize(
                        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        use_4bit=False
                    )
                    logger.info("✅ Fallback la TinyLlama reușit!")
                except Exception as fallback_error:
                    logger.error(f"❌ Și fallback-ul a eșuat: {fallback_error}")
                    raise
    
    def _prepare_text_for_analysis(self, text, max_chars=2000):
        """Pregateste textul pentru analiza"""
        if not text or len(text.strip()) < 10:
            return ""
        
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        
        if len(cleaned_text) > max_chars:
            logger.info(f"Text lung ({len(cleaned_text)} caractere), trunchiez la {max_chars}")
            
            truncate_pos = max_chars
            for i in range(max_chars - 100, max_chars):
                if i < len(cleaned_text) and cleaned_text[i] in '.!?':
                    truncate_pos = i + 1
                    break
            
            cleaned_text = cleaned_text[:truncate_pos].strip()
            if not cleaned_text.endswith(('.', '!', '?')):
                cleaned_text += "..."
            
            logger.info(f"Text trunchiat la {len(cleaned_text)} caractere")
        
        return cleaned_text
    
    def analyze_text_content(self, text, title=None):
        """Analizeaza continutul textual cu modelul upgraded"""
        if not text or len(text.strip()) < 10:
            return {
                "factuality_score": 0,
                "confidence": 0,
                "reasoning": "Text prea scurt pentru analiza.",
                "questionable_claims": []
            }
        
        analyzed_text = self._prepare_text_for_analysis(text, max_chars=2000)
        
        if not analyzed_text:
            return {
                "factuality_score": 0,
                "confidence": 0,
                "reasoning": "Nu s-a putut procesa textul pentru analiza.",
                "questionable_claims": []
            }
        
        # Prompt optimizat pentru modele mai puternice
        prompt = self._create_enhanced_factuality_prompt(analyzed_text, title)
        
        try:
            # Generare cu parametri optimizați pentru factualitate
            response = self.model_handler.generate_response(
                prompt, 
                max_new_tokens=400,  # Mai mult spațiu pentru răspuns detaliat
                temperature=0.3     # Temperatura scăzută pentru consistență
            )
            
            logger.info(f"=== RĂSPUNS MODEL UPGRADED ===")
            logger.info(f"Lungime: {len(response)} caractere")
            logger.info(f"Continut: {response}")
            logger.info(f"=== SFARSIT RĂSPUNS ===")
            
            # Parseaza raspunsul
            parsed_result = self._parse_factuality_response(response)
            
            logger.info(f"=== REZULTAT FINAL ===")
            logger.info(f"Scor: {parsed_result.get('factuality_score')}")
            logger.info(f"Incredere: {parsed_result.get('confidence')}")
            logger.info(f"=== SFARSIT REZULTAT ===")
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Eroare la analiza factualitatii: {str(e)}")
            return {
                "factuality_score": 5,
                "confidence": 3,
                "reasoning": f"Eroare la procesare: {str(e)[:100]}...",
                "questionable_claims": []
            }
    
    def _create_enhanced_factuality_prompt(self, text, title=None):
        """Prompt îmbunătățit pentru modele mai puternice"""
        title_text = f"Titlu: {title}\n\n" if title else ""
        
        # Detectează modelul pentru a adapta prompt-ul
        model_name = getattr(self.model_handler, 'model_name', 'unknown')
        
        if 'mistral' in model_name.lower():
            # Prompt optimizat pentru Mistral
            prompt = f"""[INST] Ești un expert în verificarea factualității. Analizează următorul text și evaluează-i credibilitatea.

{title_text}Text: "{text}"

Criteriile de evaluare:
1. Factualitatea informațiilor
2. Prezența surselor/dovezilor  
3. Consistența logică
4. Obiectivitatea
5. Plausibilitatea

Răspunde DOAR în format JSON:
{{
    "factuality_score": [1-10],
    "confidence": [1-10],
    "reasoning": "[explicație scurtă]",
    "questionable_claims": ["[afirmație 1]", "[afirmație 2]"]
}}

Scorul: 1-3=fals, 4-6=mixt, 7-10=credibil [/INST]"""
        
        else:
            # Prompt generic pentru alte modele
            prompt = f"""Analizează următorul text pentru factualitate și credibilitate.

{title_text}Text de analizat: "{text}"

Te rog să evaluezi textul pe criterii de factualitate, surse, logică și obiectivitate.

Răspunde în format JSON:
{{
    "factuality_score": [scor 1-10],
    "confidence": [încredere 1-10],
    "reasoning": "[explicație]",
    "questionable_claims": ["[afirmație dubioasă]"]
}}"""
        
        return prompt
    
    def _parse_factuality_response(self, response):
        """Parseaza raspunsul pentru a extrage evaluarea"""
        try:
            # Curat răspunsul
            cleaned_response = response.strip()
            
            # Încearcă să găsească JSON direct
            if cleaned_response.startswith('{') and cleaned_response.endswith('}'):
                try:
                    result = json.loads(cleaned_response)
                    logger.info("JSON direct parsat cu succes!")
                    return self._validate_and_complete_result(result)
                except json.JSONDecodeError:
                    pass
            
            # Caută JSON în răspuns cu mai multe pattern-uri
            json_patterns = [
                r'({[\s\S]*?})',
                r'```json\s*({[\s\S]*?})\s*```',
                r'```\s*({[\s\S]*?})\s*```',
                r'json\s*({[\s\S]*?})'
            ]
            
            for pattern in json_patterns:
                match = re.search(pattern, cleaned_response, re.IGNORECASE)
                if match:
                    json_str = match.group(1).strip()
                    json_str = self._clean_json_string(json_str)
                    
                    try:
                        result = json.loads(json_str)
                        logger.info(f"JSON găsit cu pattern: {pattern}")
                        return self._validate_and_complete_result(result)
                    except json.JSONDecodeError:
                        continue
            
            # Fallback: analiză text liber
            logger.warning("Nu s-a găsit JSON valid, analizez text liber")
            return self._analyze_free_text_response(response)
            
        except Exception as e:
            logger.error(f"Eroare la parsarea răspunsului: {e}")
            return self._fallback_response(response)
    
    def _clean_json_string(self, json_str):
        """Curăță JSON-ul de probleme comune"""
        # Elimină comentariile
        json_str = re.sub(r'//.*', '', json_str)
        json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)
        
        # Înlocuiește ghilimelele simple
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
        
        # Elimină virgulele finale
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str.strip()
    
    def _validate_and_complete_result(self, result):
        """Validează și completează rezultatul"""
        defaults = {
            "factuality_score": 5,
            "confidence": 5,
            "reasoning": "Analiza completata",
            "questionable_claims": []
        }
        
        for key, default_value in defaults.items():
            if key not in result:
                result[key] = default_value
        
        # Validează scorurile
        try:
            result["factuality_score"] = max(1, min(10, int(float(result["factuality_score"]))))
            result["confidence"] = max(1, min(10, int(float(result["confidence"]))))
        except (ValueError, TypeError):
            result["factuality_score"] = 5
            result["confidence"] = 5
        
        # Limitează lungimea
        if isinstance(result["reasoning"], str) and len(result["reasoning"]) > 500:
            result["reasoning"] = result["reasoning"][:500] + "..."
        
        if not isinstance(result["questionable_claims"], list):
            result["questionable_claims"] = []
        
        return result
    
    def _analyze_free_text_response(self, response):
        """Analizează răspuns în text liber"""
        text_lower = response.lower()
        
        # Caută indicatori de credibilitate
        positive_count = sum(1 for word in ['accurate', 'correct', 'true', 'reliable', 'verified'] if word in text_lower)
        negative_count = sum(1 for word in ['false', 'incorrect', 'misleading', 'questionable', 'doubtful'] if word in text_lower)
        
        if negative_count > positive_count:
            base_score = 4
            confidence = 6
        elif positive_count > negative_count:
            base_score = 7
            confidence = 7
        else:
            base_score = 5
            confidence = 5
        
        return {
            "factuality_score": base_score,
            "confidence": confidence,
            "reasoning": f"Analiza bazată pe indicatori în răspuns: {positive_count} pozitivi, {negative_count} negativi",
            "questionable_claims": []
        }
    
    def _fallback_response(self, original_response):
        """Răspuns de rezervă"""
        numbers = re.findall(r'\b(\d+)\b', original_response)
        score = 5
        if numbers:
            for num in numbers:
                num_val = int(num)
                if 1 <= num_val <= 10:
                    score = num_val
                    break
        
        return {
            "factuality_score": score,
            "confidence": 4,
            "reasoning": f"Analiză parțială: {original_response[:150]}...",
            "questionable_claims": []
        }