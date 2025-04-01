# app/llm_module/model_handler.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import logging

logger = logging.getLogger(__name__)

class ModelHandler:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelHandler, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.device = None
            cls._instance.initialized = False
        return cls._instance
    
    def initialize(self, model_id="mistralai/Mistral-7B-Instruct-v0.2", 
                  cache_dir="./model_cache", 
                  use_4bit=True):
        """Inițializează modelul și tokenizer-ul."""
        if self.initialized:
            return
        
        try:
            logger.info(f"Încărcare model {model_id}...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Utilizare dispozitiv: {self.device}")
            
            # Descarcă modelul dacă nu există local
            if not os.path.exists(os.path.join(cache_dir, model_id.split('/')[-1])):
                logger.info("Descărcare model din Hugging Face Hub...")
                snapshot_download(repo_id=model_id, local_dir=cache_dir)
            
            # Configurare pentru quantizare 4-bit pentru a reduce cerințele de memorie
            if use_4bit and self.device == "cuda":
                from bitsandbytes.nn import Linear4bit
                import transformers
                from accelerate import init_empty_weights
                
                # Încărcare tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    padding_side="left"
                )
                
                # Încărcare model cu quantizare 4-bit
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    device_map="auto",
                    load_in_4bit=True,
                    torch_dtype=torch.bfloat16
                )
            else:
                # Încărcare standard pentru CPU sau fără quantizare
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_id, 
                    cache_dir=cache_dir
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    cache_dir=cache_dir,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
                )
            
            # Setare padding token dacă nu există
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.initialized = True
            logger.info("Model încărcat cu succes.")
        except Exception as e:
            logger.error(f"Eroare la încărcarea modelului: {str(e)}")
            raise
    
    def generate_response(self, prompt, max_length=512, temperature=0.7):
        """Generează un răspuns pe baza promptului dat."""
        if not self.initialized:
            raise RuntimeError("Modelul nu a fost inițializat. Apelați metoda initialize() mai întâi.")
        
        try:
            # Formatare prompt pentru Mistral Instruct
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True)
            
            # Mutăm input-urile pe dispozitivul corect
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generăm răspunsul
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decodăm și returnăm răspunsul, eliminând promptul
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extragem doar răspunsul modelului (fără prompt)
            response = response.split("[/INST]")[-1].strip()
            
            return response
        except Exception as e:
            logger.error(f"Eroare la generarea răspunsului: {str(e)}")
            return "A apărut o eroare la procesarea textului."