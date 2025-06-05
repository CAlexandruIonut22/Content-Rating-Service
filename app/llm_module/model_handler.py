import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download, login, HfFolder
import logging

logger = logging.getLogger(__name__)

class ModelHandler:
    """Gestionează modelul LLM - folosește pattern Singleton ca să nu încarc
    modelul de mai multe ori că nu-s bogat în RAM"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelHandler, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.device = None
            cls._instance.initialized = False
            cls._instance.max_context_length = 2048  # Context maxim pentru TinyLlama
        return cls._instance
    
    def initialize(self, model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                  cache_dir="./model_cache", 
                  use_4bit=False):  # Nu e nevoie de 4-bit pentru modele mici
        """Inițializează modelul - o să dureze prima dată când rulezi"""
        if self.initialized:
            return  # nu reinițializăm dacă e deja gata
        
        try:
            logger.info(f"Începere inițializare model {model_id}...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Folosesc device: {self.device}")
            
            # Verifică dacă directorul de cache există, dacă nu, creează-l
            if not os.path.exists(cache_dir):
                logger.info(f"Creez directorul de cache: {cache_dir}")
                os.makedirs(cache_dir, exist_ok=True)
            
            # Descarcă modelul dacă nu există local
            model_folder = os.path.join(cache_dir, model_id.split('/')[-1])
            if not os.path.exists(model_folder):
                logger.info(f"Modelul nu există local, îl descarc din Hugging Face Hub...")
                logger.info(f"Asta va dura un pic, modelul are ~500MB...")
                snapshot_download(repo_id=model_id, local_dir=model_folder)
            
            # Încărcare standard pentru TinyLlama
            logger.info("Încarc tokenizer-ul...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                cache_dir=cache_dir
            )
            
            logger.info("Încarc modelul TinyLlama (mult mai mic decât Mistral)...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch.float32,  # Folosim precizie float32 pentru stabilitate
                low_cpu_mem_usage=True
            )
            
            # Setup padding token dacă nu există
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Afișează informații despre memorie dacă suntem pe GPU
            if self.device == "cuda":
                try:
                    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
                    total_in_GB = int(torch.cuda.mem_get_info()[1]/1024**3)
                    used_in_GB = total_in_GB - free_in_GB
                    logger.info(f"GPU: memorie folosită {used_in_GB}GB, liberă {free_in_GB}GB, total {total_in_GB}GB")
                except:
                    logger.info("Nu am putut obține informații despre memoria GPU")
            
            self.initialized = True
            logger.info("✅ Model încărcat cu succes! Totul e pregătit.")
        except Exception as e:
            logger.error(f"❌ A apărut o eroare la încărcarea modelului: {str(e)}")
            raise
    
    def _truncate_input(self, prompt, max_input_length=800):
        """Trunchiază input-ul pentru a se încadra în contextul modelului"""
        # Tokenizez prompt-ul pentru a vedea lungimea
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        if len(tokens) <= max_input_length:
            return prompt
        
        # Dacă e prea lung, păstrez începutul și sfârșitul
        logger.warning(f"Prompt prea lung ({len(tokens)} tokens), trunchiez la {max_input_length}")
        
        # Păstrez primele și ultimele părți
        start_tokens = tokens[:max_input_length//2]
        end_tokens = tokens[-(max_input_length//2):]
        
        # Reconstruiesc textul
        start_text = self.tokenizer.decode(start_tokens, skip_special_tokens=True)
        end_text = self.tokenizer.decode(end_tokens, skip_special_tokens=True)
        
        truncated_prompt = start_text + "\n[...text trunchiat...]\n" + end_text
        return truncated_prompt
    
    def generate_response(self, prompt, max_new_tokens=256, temperature=0.7):
        """Generează un răspuns la un prompt dat - CORECTAT pentru a evita eroarea de lungime"""
        if not self.initialized:
            raise RuntimeError("Modelul nu e inițializat! Apelează initialize() mai întâi.")
        
        try:
            # Trunchiez prompt-ul dacă e prea lung
            truncated_prompt = self._truncate_input(prompt, max_input_length=800)
            
            logger.info(f"Generez răspuns pentru prompt (primele 50 caractere): '{truncated_prompt[:50]}...'")
            
            # Tokenizare prompt cu truncation
            inputs = self.tokenizer(truncated_prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            
            # Mut input-ul pe device-ul corect (CUDA sau CPU)
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Calculez lungimea input-ului
            input_length = inputs['input_ids'].shape[1]
            logger.info(f"Lungime input: {input_length} tokens")
            
            # IMPORTANT: Folosesc max_new_tokens în loc de max_length
            with torch.no_grad():  # fără calcul de gradient, doar inferență
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,  # SCHIMBAT: folosesc max_new_tokens
                    temperature=temperature,  # cât de "creativ" să fie
                    do_sample=True,  # sampling pentru diversitate
                    top_p=0.95,  # nucleus sampling (mai natural)
                    repetition_penalty=1.2,  # să nu repete același lucru
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decodez output-ul în text
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Pentru TinyLlama cu format de chat, extrag doar partea nouă
            if "<|im_start|>assistant" in full_response:
                # Găsesc ultima apariție a tag-ului assistant
                parts = full_response.split("<|im_start|>assistant")
                if len(parts) > 1:
                    response = parts[-1].strip()
                    # Elimin tag-ul de închidere dacă există
                    if response.startswith("\n"):
                        response = response[1:]
                    if "<|im_end|>" in response:
                        response = response.split("<|im_end|>")[0]
                else:
                    response = full_response
            else:
                # Dacă nu găsesc tag-urile, încerc să elimin prompt-ul
                response = full_response.replace(truncated_prompt, "").strip()
                if not response:
                    response = full_response
            
            logger.info(f"Răspuns generat cu succes: {len(response)} caractere")
            logger.info(f"Răspuns extras: {response[:100]}...")
            return response
        except Exception as e:
            logger.error(f"Eroare la generarea răspunsului: {str(e)}")
            return f"A apărut o eroare la procesarea textului: {str(e)[:100]}... Te rog încearcă din nou cu un text mai scurt."