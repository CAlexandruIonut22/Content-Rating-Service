# -*- coding: utf-8 -*-
from .model_handler import ModelHandler
import logging
import json
import re

logger = logging.getLogger(__name__)

class FactualityChecker:
    """Clasa pentru verificarea factualității textului folosind modelul LLM"""
    
    def __init__(self):
        """Inițializează checker-ul de factualitate"""
        self.model_handler = ModelHandler()
        if not self.model_handler.initialized:
            logger.info("Initializez modelul TinyLlama din FactualityChecker...")
            self.model_handler.initialize()
    
    def _prepare_text_for_analysis(self, text, max_chars=1500):
        """Pregateste textul pentru analiza, limitandu-l la o lungime rezonabila"""
        if not text or len(text.strip()) < 10:
            return ""
        
        # Curat textul de caractere problematice
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        
        # Daca textul e prea lung, il trunchiez 
        if len(cleaned_text) > max_chars:
            logger.info(f"Text lung ({len(cleaned_text)} caractere), trunchiez la {max_chars}")
            
            # Incerc sa gasesc o intrerupere naturala (punct, punct si virgula)
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
        """Analizeaza continutul textual si returneaza evaluarea factualitatii"""
        if not text or len(text.strip()) < 10:
            return {
                "factuality_score": 0,
                "confidence": 0,
                "reasoning": "Text prea scurt pentru analiza.",
                "questionable_claims": []
            }
        
        # Pregatesc textul pentru analiza
        analyzed_text = self._prepare_text_for_analysis(text, max_chars=1200)
        
        if not analyzed_text:
            return {
                "factuality_score": 0,
                "confidence": 0,
                "reasoning": "Nu s-a putut procesa textul pentru analiza.",
                "questionable_claims": []
            }
        
        # Creaza promptul pentru analiza de factualitate, adaptat pentru TinyLlama
        prompt = self._create_factuality_prompt(analyzed_text, title)
        
        try:
            # Obtine raspunsul de la model cu parametri optimizati
            response = self.model_handler.generate_response(
                prompt, 
                max_new_tokens=300,  # Redus pentru raspuns mai focalizat
                temperature=0.2  # Temperatura mai mica pentru consistenta
            )
            
            # DEBUG: Afisez raspunsul raw pentru debugging
            logger.info(f"=== RASPUNS RAW DE LA MODEL ===")
            logger.info(f"Lungime: {len(response)} caractere")
            logger.info(f"Continut: {response}")
            logger.info(f"=== SFARSIT RASPUNS RAW ===")
            
            # Parseaza raspunsul pentru a extrage evaluarea structurata
            parsed_result = self._parse_factuality_response(response)
            
            # DEBUG: Afisez rezultatul final
            logger.info(f"=== REZULTAT FINAL PARSAT ===")
            logger.info(f"Scor: {parsed_result.get('factuality_score')}")
            logger.info(f"Incredere: {parsed_result.get('confidence')}")
            logger.info(f"Rationament: {parsed_result.get('reasoning')}")
            logger.info(f"=== SFARSIT REZULTAT ===")
            
            return parsed_result
        except Exception as e:
            logger.error(f"Eroare la analiza factualitatii: {str(e)}")
            return {
                "factuality_score": 5,  # Valoare neutra in caz de eroare
                "confidence": 3,
                "reasoning": f"Eroare la procesare (text posibil prea lung): {str(e)[:100]}...",
                "questionable_claims": []
            }
    
    def _create_factuality_prompt(self, text, title=None):
        """Creaza un prompt pentru analiza factualitatii, adaptat pentru TinyLlama"""
        title_text = f"Title: {title}\n\n" if title else ""
        
        # Prompt în engleză și mai direct pentru TinyLlama (pare să funcționeze mai bine în engleză)
        prompt = f"""<|im_start|>system
You are a fact-checking assistant. You must respond ONLY with valid JSON, nothing else.
<|im_end|>
<|im_start|>user
Rate this text for factual accuracy from 1-10.

{title_text}Text: "{text[:800]}..."

Respond with ONLY this JSON format, no other text:
{{
"factuality_score": 7,
"confidence": 8,
"reasoning": "Brief explanation",
"questionable_claims": []
}}
<|im_end|>
<|im_start|>assistant
{{"""
        
        return prompt
    
    def _parse_factuality_response(self, response):
        """Parseaza raspunsul LLM pentru a extrage evaluarea structurata"""
        try:
            # Curat raspunsul de caractere problematice
            cleaned_response = response.strip()
            
            # Daca raspunsul incepe cu { si se termina cu } -> probabil e JSON
            if cleaned_response.startswith('{') and cleaned_response.endswith('}'):
                try:
                    result = json.loads(cleaned_response)
                    logger.info("JSON direct parsat cu succes!")
                    return self._validate_and_complete_result(result)
                except json.JSONDecodeError:
                    logger.warning("JSON aparent valid dar cu erori de sintaxa")
            
            # Incerc sa gasesc JSON-ul in raspuns - mai multe strategii
            json_patterns = [
                r'({[\s\S]*?})',  # Primul JSON gasit
                r'```json\s*({[\s\S]*?})\s*```',  # JSON intre markere
                r'```\s*({[\s\S]*?})\s*```',  # JSON intre triple backticks
                r'json\s*({[\s\S]*?})',  # Dupa cuvantul "json"
            ]
            
            json_str = None
            for pattern in json_patterns:
                match = re.search(pattern, cleaned_response, re.IGNORECASE)
                if match:
                    json_str = match.group(1).strip()
                    logger.info(f"JSON gasit cu pattern: {pattern}")
                    logger.info(f"JSON extras: {json_str}")
                    break
            
            if json_str:
                # Curat JSON-ul
                json_str = self._clean_json_string(json_str)
                
                try:
                    result = json.loads(json_str)
                    logger.info("JSON parsat cu succes!")
                    return self._validate_and_complete_result(result)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON invalid dupa curatare: {e}")
                    # Incerc repararea JSON-ului
                    repaired_json = self._repair_json(json_str)
                    if repaired_json:
                        try:
                            result = json.loads(repaired_json)
                            logger.info("JSON reparat si parsat cu succes!")
                            return self._validate_and_complete_result(result)
                        except:
                            logger.warning("Nici JSON-ul reparat nu functioneaza")
            
            # Daca nu gasesc JSON valid, analizez textul liber pentru informatii
            logger.warning("Nu s-a gasit JSON, analizez textul liber pentru informatii")
            return self._analyze_free_text_response(response)
                
        except Exception as e:
            logger.error(f"Eroare la parsarea raspunsului: {str(e)}")
            return self._fallback_response(response)
    
    def _analyze_free_text_response(self, response):
        """Analizeaza un raspuns in text liber si extrage informatii de factualitate"""
        logger.info("Analizez raspunsul ca text liber...")
        
        # Analiza continutului pentru a determina factualitatea
        text_lower = response.lower()
        
        # Cautare indicatori de credibilitate
        positive_indicators = [
            'accurate', 'correct', 'factual', 'true', 'reliable', 'verified', 
            'documented', 'supported', 'evidence', 'confirmed', 'valid'
        ]
        
        negative_indicators = [
            'false', 'incorrect', 'misleading', 'inaccurate', 'unverified', 
            'questionable', 'doubtful', 'unsupported', 'claim', 'alleged'
        ]
        
        neutral_indicators = [
            'however', 'but', 'although', 'while', 'mixed', 'partial', 
            'some', 'unclear', 'limited', 'no specific details'
        ]
        
        # Contorizez indicatorii
        positive_count = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in text_lower)
        neutral_count = sum(1 for indicator in neutral_indicators if indicator in text_lower)
        
        # Determin scorul bazat pe indicatori
        if negative_count > positive_count:
            base_score = 4  # Continut suspect
            confidence = 6
        elif positive_count > negative_count:
            base_score = 7  # Continut credibil
            confidence = 7
        else:
            base_score = 5  # Continut mixt
            confidence = 5
        
        # Ajustez scorul bazat pe prezenta anumitor fraze
        if 'no specific details' in text_lower or 'unclear' in text_lower:
            base_score = max(4, base_score - 1)
            confidence = max(4, confidence - 1)
        
        if 'support' in text_lower and 'evidence' in text_lower:
            base_score = min(8, base_score + 1)
            confidence = min(8, confidence + 1)
        
        # Creez rationamentul bazat pe analiza
        reasoning_parts = []
        
        if positive_count > 0:
            reasoning_parts.append(f"Textul contine {positive_count} indicatori pozitivi de credibilitate")
        
        if negative_count > 0:
            reasoning_parts.append(f"S-au identificat {negative_count} indicatori de precautie")
        
        if neutral_count > 0:
            reasoning_parts.append(f"Exista {neutral_count} indicatori de neutralitate/incertitudine")
        
        # Adaugez observatii specifice
        if 'no specific details' in text_lower:
            reasoning_parts.append("Lipsesc detalii specifice care sa poata fi verificate")
        
        if 'support' in text_lower:
            reasoning_parts.append("Se mentioneaza suport sau sustinere pentru anumite pozitii")
        
        if 'opposition' in text_lower:
            reasoning_parts.append("Se mentioneaza existenta unei opozitii")
        
        reasoning = ". ".join(reasoning_parts) if reasoning_parts else "Analiza bazata pe continutul general al textului"
        
        # Caut afirmatii problematice
        questionable_claims = []
        
        # Identific propozitii care par sa faca afirmatii fara dovezi
        sentences = re.split(r'[.!?]', response)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:
                # Verific daca propozitia contine afirmatii care ar trebui verificate
                if any(phrase in sentence.lower() for phrase in [
                    'announces plans', 'claims', 'states', 'reports', 'alleges'
                ]):
                    if len(sentence) < 150:  # Nu adaug propozitii prea lungi
                        questionable_claims.append(sentence)
        
        # Limitez la maxim 3 afirmatii
        questionable_claims = questionable_claims[:3]
        
        result = {
            "factuality_score": base_score,
            "confidence": confidence,
            "reasoning": reasoning[:400] + ("..." if len(reasoning) > 400 else ""),
            "questionable_claims": questionable_claims
        }
        
        logger.info(f"Analiza text liber completata: scor={base_score}, incredere={confidence}")
        return result
    
    def _clean_json_string(self, json_str):
        """Curata string-ul JSON de probleme comune"""
        # Elimina comentariile
        json_str = re.sub(r'//.*', '', json_str)
        json_str = re.sub(r'/\*[\s\S]*?\*/', '', json_str)
        
        # Inlocuieste ghilimelele simple cu duble
        json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)  # Chei
        json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)  # Valori
        
        # Elimina virgulele finale
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Elimina caracterele de control
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
        
        return json_str.strip()
    
    def _repair_json(self, json_str):
        """Incearca sa repare un JSON partial sau corupt"""
        try:
            # Incerc sa completez JSON-ul daca e partial
            if not json_str.strip().endswith('}'):
                # Verific daca lipseste }
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                if open_braces > close_braces:
                    json_str += '}' * (open_braces - close_braces)
            
            # Incerc sa adaug campurile lipsa cu valori default
            if '"factuality_score"' not in json_str:
                json_str = json_str.replace('{', '{"factuality_score": 5,', 1)
            
            if '"confidence"' not in json_str:
                json_str = json_str.replace('"factuality_score"', '"factuality_score"').replace('5,', '5, "confidence": 5,', 1)
            
            if '"reasoning"' not in json_str:
                json_str = json_str.replace('}', ', "reasoning": "Analiza completata", "questionable_claims": []}')
            
            return json_str
        except:
            return None
    
    def _validate_and_complete_result(self, result):
        """Valideaza si completeaza rezultatul"""
        # Campurile necesare cu valori default
        defaults = {
            "factuality_score": 5,
            "confidence": 5,
            "reasoning": "Analiza completata de TinyLlama",
            "questionable_claims": []
        }
        
        # Completeaza campurile lipsa
        for key, default_value in defaults.items():
            if key not in result:
                result[key] = default_value
        
        # Valideaza scorurile
        try:
            result["factuality_score"] = max(1, min(10, int(float(result["factuality_score"]))))
            result["confidence"] = max(1, min(10, int(float(result["confidence"]))))
        except (ValueError, TypeError):
            result["factuality_score"] = 5
            result["confidence"] = 5
        
        # Limiteaza lungimea rationamentului
        if isinstance(result["reasoning"], str) and len(result["reasoning"]) > 500:
            result["reasoning"] = result["reasoning"][:500] + "..."
        
        # Asigura ca questionable_claims e o lista
        if not isinstance(result["questionable_claims"], list):
            result["questionable_claims"] = []
        
        return result
    
    def _manual_parse_response(self, response):
        """Extrage manual informatiile din raspuns cand JSON-ul nu functioneaza"""
        try:
            logger.info("Incep extragerea manuala...")
            
            # Caut scoruri in text cu mai multe pattern-uri
            score_patterns = [
                r'factuality_score["\s:]*(\d+)',
                r'scor[^:]*?[:\s]+(\d+)',
                r'factualitate[^:]*?[:\s]+(\d+)',
                r'nota[^:]*?[:\s]+(\d+)',
                r'(\d+)/10',
                r'(\d+)\s*din\s*10'
            ]
            
            confidence_patterns = [
                r'confidence["\s:]*(\d+)',
                r'incredere[^:]*?[:\s]+(\d+)',
                r'sigur[^:]*?[:\s]+(\d+)',
                r'certitudine[^:]*?[:\s]+(\d+)'
            ]
            
            # Extrag scorul
            score = 5
            for pattern in score_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        score = int(match.group(1))
                        if 1 <= score <= 10:
                            logger.info(f"Scor gasit: {score} cu pattern: {pattern}")
                            break
                    except ValueError:
                        continue
            
            # Extrag increderea
            confidence = 5
            for pattern in confidence_patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        confidence = int(match.group(1))
                        if 1 <= confidence <= 10:
                            logger.info(f"Incredere gasita: {confidence} cu pattern: {pattern}")
                            break
                    except ValueError:
                        continue
            
            # Extrag rationamentul - caut textul cel mai lung care pare o explicatie
            reasoning_text = "TinyLlama a analizat textul dar nu a fost gasit un rationament clar."
            
            # Caut propozitii care par sa fie explicatii
            sentences = re.split(r'[.!?]\s+', response)
            for sentence in sentences:
                if len(sentence) > 50 and any(word in sentence.lower() for word in 
                    ['pentru', 'deoarece', 'fiindca', 'prin urmare', 'astfel', 'se pare', 'considera']):
                    reasoning_text = sentence.strip()
                    if not reasoning_text.endswith(('.', '!', '?')):
                        reasoning_text += "."
                    break
            
            result = {
                "factuality_score": score,
                "confidence": confidence,
                "reasoning": reasoning_text[:300],
                "questionable_claims": []
            }
            
            logger.info(f"Extragere manuala completata: scor={score}, incredere={confidence}")
            return result
            
        except Exception as e:
            logger.error(f"Eroare la extragerea manuala: {e}")
            return self._fallback_response(response)
    
    def _fallback_response(self, original_response):
        """Raspuns de rezerva cand tot restul esueaza"""
        # Incerc sa gasesc macar un numar in raspuns pentru scor
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
            "reasoning": f"TinyLlama a analizat textul dar raspunsul nu a putut fi interpretat complet. Raspuns partial: {original_response[:150]}...",
            "questionable_claims": []
        }