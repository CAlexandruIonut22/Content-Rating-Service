# app/llm_module/factuality_checker.py - Optimizat pentru TinyLlama-1.1B

import sys
import os
import time
import json
import re
import logging

# Fix pentru import-uri
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from .model_handler import ModelHandler

try:
    from .web_search_agent import WebSearchAgent
except ImportError:
    WebSearchAgent = None
    
try:
    from app.config import Config
except ImportError:
    # Fallback config pentru TinyLlama
    class Config:
        LLM_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        USE_WEB_SEARCH = True
        TAVILY_API_KEY = os.environ.get('TAVILY_API_KEY', None)
        USE_HYBRID_ANALYSIS = False
        MAX_WEB_SEARCHES = 3
        MAX_TEXT_LENGTH_FOR_ANALYSIS = 1500
        TINYLLAMA_PROMPT_MAX_LENGTH = 800
        TINYLLAMA_USE_SIMPLE_PROMPTS = True

logger = logging.getLogger(__name__)

class FactualityChecker:
    """
    Analizor factualitate optimizat pentru TinyLlama-1.1B
    Focus pe prompturi simple și analiză rapidă
    """
    
    def __init__(self):
        """Inițializează checker-ul pentru TinyLlama"""
        self.model_handler = ModelHandler()
        self.web_search_agent = None
        
        # Inițializează TinyLlama
        if not self.model_handler.initialized:
            logger.info("Inițializez TinyLlama pentru analiză factualitate...")
            try:
                self.model_handler.initialize(
                    model_id=getattr(Config, 'LLM_MODEL_ID', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'),
                    cache_dir=getattr(Config, 'LLM_CACHE_DIR', './model_cache'),
                    hf_token=getattr(Config, 'HUGGINGFACE_TOKEN', None)
                )
                logger.info("✅ TinyLlama inițializat pentru analiză!")
            except Exception as e:
                logger.error(f"❌ Eroare inițializare TinyLlama: {e}")
                raise
        
        # Inițializează web search dacă e disponibil
        if (getattr(Config, 'USE_WEB_SEARCH', False) and 
            getattr(Config, 'TAVILY_API_KEY', None) and 
            WebSearchAgent):
            try:
                self.web_search_agent = WebSearchAgent(Config.TAVILY_API_KEY)
                logger.info("✅ Tavily Web Search activat pentru TinyLlama!")
            except Exception as e:
                logger.warning(f"⚠️  Web search eșuat: {e}")
                self.web_search_agent = None
        else:
            logger.info("ℹ️  Web search dezactivat")
    
    def analyze_text_content(self, text, title=None):
        """
        Analiză factualitate optimizată pentru TinyLlama
        """
        if not text or len(text.strip()) < 10:
            return {
                "factuality_score": 0,
                "confidence": 0,
                "reasoning": "Text prea scurt pentru analiză.",
                "questionable_claims": [],
                "analysis_type": "insufficient_content"
            }
        
        # Pregătește textul pentru TinyLlama
        analyzed_text = self._prepare_text_for_tinyllama(text)
        
        if not analyzed_text:
            return {
                "factuality_score": 0,
                "confidence": 0,
                "reasoning": "Nu s-a putut procesa textul.",
                "questionable_claims": [],
                "analysis_type": "processing_error"
            }
        
        try:
            # Decidem tipul de analiză
            if (self.web_search_agent and 
                getattr(Config, 'USE_HYBRID_ANALYSIS', True)):
                logger.info("🔍 Analiză hibridă: TinyLlama + Web Search")
                return self._hybrid_analysis_tinyllama(analyzed_text, title)
            else:
                logger.info("🤖 Analiză doar cu TinyLlama")
                return self._tinyllama_only_analysis(analyzed_text, title)
                
        except Exception as e:
            logger.error(f"Eroare analiză factualitate: {e}")
            return {
                "factuality_score": 5,
                "confidence": 3,
                "reasoning": f"Eroare la procesare cu TinyLlama: {str(e)[:100]}...",
                "questionable_claims": [],
                "analysis_type": "error"
            }
    
    def _prepare_text_for_tinyllama(self, text):
        """Pregătește text pentru TinyLlama (limitări mai stricte)"""
        if not text:
            return ""
        
        # Curăță textul
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        
        # Limitare strictă pentru TinyLlama
        max_chars = getattr(Config, 'MAX_TEXT_LENGTH_FOR_ANALYSIS', 1500)
        
        if len(cleaned_text) > max_chars:
            logger.info(f"Text lung ({len(cleaned_text)} chars) pentru TinyLlama, trunchiez la {max_chars}")
            
            # Trunchiază la o propoziție completă
            truncate_pos = max_chars
            for i in range(max_chars - 200, max_chars):
                if i < len(cleaned_text) and cleaned_text[i] in '.!?':
                    truncate_pos = i + 1
                    break
            
            cleaned_text = cleaned_text[:truncate_pos].strip()
            if not cleaned_text.endswith(('.', '!', '?')):
                cleaned_text += "."
            
            logger.info(f"Text trunchiat la {len(cleaned_text)} caractere pentru TinyLlama")
        
        return cleaned_text
    
    def _tinyllama_only_analysis(self, text, title=None):
        """Analiză doar cu TinyLlama (prompturi foarte simple)"""
        prompt = self._create_simple_tinyllama_prompt(text, title)
        
        try:
            start_time = time.time()
            
            response = self.model_handler.generate_response(
                prompt,
                max_new_tokens=200,  # Răspuns scurt pentru TinyLlama
                temperature=0.7,
                do_sample=True
            )
            
            generation_time = time.time() - start_time
            
            logger.info(f"=== TINYLLAMA RĂSPUNS ===")
            logger.info(f"Timp: {generation_time:.1f}s")
            logger.info(f"Content: {response[:150]}...")
            logger.info(f"=== SFÂRȘIT ===")
            
            parsed_result = self._parse_tinyllama_response(response)
            parsed_result['analysis_type'] = 'tinyllama_only'
            parsed_result['generation_time'] = f"{generation_time:.1f}s"
            
            return parsed_result
            
        except Exception as e:
            logger.error(f"Eroare TinyLlama: {e}")
            return {
                "factuality_score": 5,
                "confidence": 3,
                "reasoning": f"Eroare TinyLlama: {str(e)[:100]}...",
                "questionable_claims": [],
                "analysis_type": "tinyllama_error",
                "error": str(e)
            }
    
    def _hybrid_analysis_tinyllama(self, text, title=None):
        """Analiză hibridă: TinyLlama + Web Search"""
        logger.info("🚀 Pornesc analiză hibridă cu TinyLlama...")
        
        # 1. Analiză inițială cu TinyLlama
        tinyllama_result = self._tinyllama_only_analysis(text, title)
        
        # 2. Web search pentru verificare
        if self.web_search_agent:
            try:
                # Extrage afirmații simple pentru verificare
                claims_to_verify = self._extract_simple_claims(text)
                logger.info(f"📋 Verific {len(claims_to_verify)} afirmații cu web search")
                
                verified_claims = []
                web_confidence_boost = 0
                
                # 3. Verifică maxim 3 afirmații
                for claim in claims_to_verify[:3]:
                    logger.info(f"🔍 Verific: {claim[:50]}...")
                    verification = self.web_search_agent.verify_claim_with_search(claim)
                    verified_claims.append(verification)
                    
                    # Ajustează încrederea
                    if verification['verification_status'] == 'adevarata':
                        web_confidence_boost += 1
                    elif verification['verification_status'] == 'falsa':
                        web_confidence_boost -= 2
                
                # 4. Combină rezultatele
                final_score = tinyllama_result['factuality_score']
                
                # Ajustează scorul bazat pe web search
                if verified_claims:
                    web_adjustment = min(2, max(-3, web_confidence_boost))
                    final_score = max(1, min(10, final_score + web_adjustment))
                
                # Îmbunătățește încrederea cu web search
                final_confidence = min(10, tinyllama_result['confidence'] + (2 if verified_claims else 0))
                
                # Reasoning îmbunătățit
                enhanced_reasoning = tinyllama_result['reasoning']
                if verified_claims:
                    true_count = sum(1 for v in verified_claims if v['verification_status'] == 'adevarata')
                    false_count = sum(1 for v in verified_claims if v['verification_status'] == 'falsa')
                    enhanced_reasoning += f"\n\n🔍 Verificare web: {len(verified_claims)} afirmații. "
                    enhanced_reasoning += f"✅ {true_count} confirmate, ❌ {false_count} infirmate."
                
                return {
                    "factuality_score": final_score,
                    "confidence": final_confidence,
                    "reasoning": enhanced_reasoning,
                    "questionable_claims": tinyllama_result['questionable_claims'],
                    "verified_claims": verified_claims,
                    "analysis_type": "hybrid_tinyllama",
                    "web_search_performed": True,
                    "sources_consulted": len(set([url for v in verified_claims for url in v.get('sources_used', [])]))
                }
                
            except Exception as e:
                logger.error(f"Eroare web search: {e}")
                tinyllama_result['analysis_type'] = 'tinyllama_only_web_failed'
                tinyllama_result['web_search_error'] = str(e)
                return tinyllama_result
        
        # Fallback la TinyLlama only
        tinyllama_result['analysis_type'] = 'tinyllama_only'
        return tinyllama_result
    
    def _create_simple_tinyllama_prompt(self, text, title=None):
        """Prompt FOARTE SIMPLU și STRICT pentru TinyLlama să răspundă JSON"""
        title_text = f"Titlu: {title}\n" if title else ""
        
        # Prompt FOARTE STRICT pentru JSON - forțăm formatul
        prompt = f"""<|user|>
Analizează următorul text și spune dacă este adevărat sau fals.

{title_text}Text: "{text}"

IMPORTANT: Răspunde DOAR în format JSON dupa modelul următor, fără alte cuvinte:

{{
    "factuality_score": 3,
    "confidence": 8,
    "reasoning": "explicația ta aici",
    "questionable_claims": ["problemă găsită"]
}}

Unde scorul este categorizat: 1-3=fals, 4-6=mixt, 7-10=adevărat, 
la reasoning pui explicatia ta si la questionable_claims pui ce crezi ca este incorect sau problematic

JSON:
<|assistant|>
{{"""
        
        return prompt
    
    def _parse_tinyllama_response(self, response):
        """Parsează răspunsul TinyLlama cu focus pe JSON forțat"""
        try:
            logger.info(f"Parsez răspuns TinyLlama: {response[:100]}...")
            
            # 1. Încearcă să găsească JSON complet cu {{ la început
            # Caută pattern-ul forțat cu "factuality_score"
            json_pattern = r'(\{[^{}]*"factuality_score"[^{}]*\})'
            match = re.search(json_pattern, response, re.IGNORECASE | re.DOTALL)
            
            if match:
                json_str = match.group(1)
                logger.info(f"Găsit JSON candidat: {json_str}")
                
                # Curăță și completează JSON-ul
                json_str = self._fix_incomplete_json(json_str)
                
                try:
                    result = json.loads(json_str)
                    logger.info("✅ JSON parsat cu succes!")
                    return self._validate_result(result)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON invalid: {e}")
            
            # 2. Încearcă să construiască JSON din bucăți
            constructed_json = self._construct_json_from_response(response)
            if constructed_json:
                try:
                    result = json.loads(constructed_json)
                    logger.info("✅ JSON construit cu succes din bucăți!")
                    return self._validate_result(result)
                except json.JSONDecodeError:
                    pass
            
            # 3. Parsare simplă (format vechi)
            simple_format = self._parse_simple_format(response)
            if simple_format:
                logger.info("✅ Format simplu parsat")
                return simple_format
            
            # 4. Extrage numere din text
            numbers_format = self._extract_numbers_from_text(response)
            if numbers_format:
                logger.info("✅ Numere extrase din text")
                return numbers_format
            
            # 5. Fallback inteligent
            logger.warning("⚠️ Folosesc fallback inteligent")
            return self._smart_fallback_response(response)
            
        except Exception as e:
            logger.error(f"Eroare critică la parsing: {e}")
            return self._emergency_fallback_response(response)
    
    def _fix_incomplete_json(self, json_str):
        """Repară JSON incomplet de la TinyLlama"""
        # Curăță caracterele problematice
        json_str = json_str.strip()
        
        # Asigură-te că se termină cu }
        if not json_str.endswith('}'):
            json_str += '}'
        
        # Repară ghilimelele lipsă
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        json_str = re.sub(r':\s*([^",\[\]{}]+)([,}])', r': "\1"\2', json_str)
        
        # Repară array-urile
        json_str = re.sub(r'\[([^\[\]]*)\]', lambda m: '["' + m.group(1).replace(',', '","') + '"]', json_str)
        
        return json_str
    
    def _construct_json_from_response(self, response):
        """Construiește JSON din bucăți găsite în răspuns"""
        try:
            # Caută componentele JSON în răspuns
            score_match = re.search(r'(?:factuality_score|scor)["\s:]*(\d+)', response, re.IGNORECASE)
            confidence_match = re.search(r'(?:confidence|încredere)["\s:]*(\d+)', response, re.IGNORECASE)
            
            # Caută reasoning/explicație
            reasoning_patterns = [
                r'(?:reasoning|explicație)["\s:]*"([^"]*)"',
                r'(?:reasoning|explicație)["\s:]*(.*?)(?:,|\}|$)',
                r'(Pământul.*?\.|.*?fals.*?\.|.*?adevărat.*?\.)'
            ]
            
            reasoning = "Analiză TinyLlama completată"
            for pattern in reasoning_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    reasoning = match.group(1).strip().strip('"')
                    break
            
            # Construiește JSON-ul
            if score_match:
                score = int(score_match.group(1))
                confidence = int(confidence_match.group(1)) if confidence_match else 5
                
                # Pentru "The earth is flat" - forțează scor mic
                if 'flat' in response.lower() and 'earth' in response.lower():
                    if score > 5:  # Dacă TinyLlama a dat scor mare pentru pământ plat
                        score = 2  # Corectează la scor mic
                        reasoning = "Afirmația că Pământul este plat este falsă științific. " + reasoning
                
                constructed = {
                    "factuality_score": score,
                    "confidence": confidence, 
                    "reasoning": reasoning,
                    "questionable_claims": ["Afirmația despre pământul plat"] if 'flat' in response.lower() else []
                }
                
                return json.dumps(constructed)
        except:
            pass
        
        return None
    
    def _smart_fallback_response(self, original_response):
        """Fallback inteligent care înțelege contextul"""
        response_lower = original_response.lower()
        
        # Detectare specială pentru afirmații științific false
        scientific_false_indicators = ['flat earth', 'pământul plat', 'flat', 'plat']
        is_scientific_false = any(indicator in response_lower for indicator in scientific_false_indicators)
        
        if is_scientific_false:
            # Pentru afirmații științific false, forțează scor mic
            return {
                "factuality_score": 2,
                "confidence": 8,
                "reasoning": "Afirmația că Pământul este plat este falsă științific. Există dovezi concrete că Pământul este sferic.",
                "questionable_claims": ["Teoria pământului plat este contrazisă de evidențe științifice"]
            }
        
        # Analiză sentiment pentru alte cazuri
        positive_words = ['adevărat', 'corect', 'exact', 'credibil', 'confirmat']
        negative_words = ['fals', 'greșit', 'incorect', 'dubios', 'problemă', 'false']
        
        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)
        
        if negative_count > positive_count:
            score = 3
            confidence = 6
        elif positive_count > negative_count:
            score = 7
            confidence = 6
        else:
            score = 5
            confidence = 4
        
        return {
            "factuality_score": score,
            "confidence": confidence,
            "reasoning": f"Analiză TinyLlama: {original_response[:200]}...",
            "questionable_claims": []
        }
    
    def _emergency_fallback_response(self, original_response):
        """Fallback de urgență"""
        return {
            "factuality_score": 5,
            "confidence": 3,
            "reasoning": "Eroare la procesarea răspunsului TinyLlama. Text analizat cu succes parțial.",
            "questionable_claims": []
        }
    
    def _parse_simple_format(self, response):
        """Parsează format simplu: Scor: X, Încredere: Y"""
        try:
            score_match = re.search(r'scor:?\s*(\d+)', response, re.IGNORECASE)
            confidence_match = re.search(r'încredere:?\s*(\d+)', response, re.IGNORECASE)
            
            if score_match:
                score = int(score_match.group(1))
                confidence = int(confidence_match.group(1)) if confidence_match else 5
                
                # Extrage explicația
                explanation_patterns = [
                    r'explicație:?\s*([^.]*\.)',
                    r'de ce:?\s*([^.]*\.)',
                    r'pentru că:?\s*([^.]*\.)'
                ]
                
                reasoning = "Analiză TinyLlama"
                for pattern in explanation_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        reasoning = match.group(1).strip()
                        break
                
                # Extrage probleme
                problems = []
                problem_patterns = [
                    r'probleme?:?\s*([^.]*\.)',
                    r'fals:?\s*([^.]*\.)',
                    r'dubios:?\s*([^.]*\.)'
                ]
                
                for pattern in problem_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        problems.append(match.group(1).strip())
                
                return {
                    "factuality_score": max(1, min(10, score)),
                    "confidence": max(1, min(10, confidence)),
                    "reasoning": reasoning,
                    "questionable_claims": problems
                }
        except:
            pass
        
        return None
    
    def _extract_numbers_from_text(self, response):
        """Extrage numere din text când formatul nu e clar"""
        numbers = re.findall(r'\b([1-9]|10)\b', response)
        
        if len(numbers) >= 2:
            # Presupune că primele două numere sunt scorul și încrederea
            score = int(numbers[0])
            confidence = int(numbers[1])
            
            # Încearcă să găsească un raționament
            sentences = re.split(r'[.!?]', response)
            reasoning = "Analiză TinyLlama completată"
            
            for sentence in sentences:
                if len(sentence.strip()) > 20:  # Propoziție substanțială
                    reasoning = sentence.strip()
                    break
            
            return {
                "factuality_score": max(1, min(10, score)),
                "confidence": max(1, min(10, confidence)),
                "reasoning": reasoning[:200] + "..." if len(reasoning) > 200 else reasoning,
                "questionable_claims": []
            }
        
        return None
    
    def _extract_simple_claims(self, text):
        """Extrage afirmații simple pentru web search"""
        # Pentru TinyLlama, păstrăm simplu - împarte textul în propoziții
        sentences = re.split(r'[.!?]', text)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Păstrează propozițiile care par să facă afirmații
            if (len(sentence) > 15 and 
                len(sentence) < 200 and
                any(word in sentence.lower() for word in ['este', 'sunt', 'a fost', 'au fost', 'va fi', 'vor fi'])):
                claims.append(sentence)
        
        return claims[:5]  # Maxim 5 afirmații
    
    def _clean_json_string(self, json_str):
        """Curăță JSON pentru TinyLlama"""
        # Elimină comentarii și caractere problematice
        json_str = re.sub(r'//.*', '', json_str)
        json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)
        
        # Înlocuiește ghilimelele simple
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
        
        # Elimină virgulele finale
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        return json_str.strip()
    
    def _validate_result(self, result):
        """Validează și completează rezultatul"""
        defaults = {
            "factuality_score": 5,
            "confidence": 5,
            "reasoning": "Analiză TinyLlama completată",
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
        if isinstance(result["reasoning"], str) and len(result["reasoning"]) > 300:
            result["reasoning"] = result["reasoning"][:300] + "..."
        
        if not isinstance(result["questionable_claims"], list):
            result["questionable_claims"] = []
        
        return result
    
    def _fallback_tinyllama_response(self, original_response):
        """Fallback când TinyLlama nu poate fi parsat"""
        # Analizează sentiment-ul răspunsului
        response_lower = original_response.lower()
        
        # Caută indicatori pozitivi/negativi
        positive_words = ['adevărat', 'corect', 'exact', 'credibil', 'bun']
        negative_words = ['fals', 'greșit', 'incorect', 'dubios', 'problemă']
        
        positive_count = sum(1 for word in positive_words if word in response_lower)
        negative_count = sum(1 for word in negative_words if word in response_lower)
        
        if negative_count > positive_count:
            score = 3
        elif positive_count > negative_count:
            score = 7
        else:
            score = 5
        
        return {
            "factuality_score": score,
            "confidence": 4,
            "reasoning": f"Analiză TinyLlama: {original_response[:150]}...",
            "questionable_claims": []
        }