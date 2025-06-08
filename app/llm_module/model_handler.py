import os
import sys
import torch
import time
import re  # ADĂUGAT pentru regex în clean_tinyllama_response
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import logging

# Fix pentru import-uri
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

logger = logging.getLogger(__name__)

class ModelHandler:
    """
    Model Handler optimizat specific pentru TinyLlama-1.1B
    Focus pe viteză și eficiență pentru hardware modest
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelHandler, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.device = None
            cls._instance.initialized = False
            cls._instance.model_name = None
            cls._instance.is_tinyllama = False
            cls._instance.generation_stats = {"total_generations": 0, "total_time": 0}
        return cls._instance
    
    def initialize(self, 
                  model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                  cache_dir="./model_cache", 
                  use_4bit=False,
                  hf_token=None):
        """
        Inițializează TinyLlama cu setări optimizate
        """
        if self.initialized:
            logger.info(f"Model {self.model_name} deja inițializat")
            return
        
        try:
            logger.info(f"🚀 Inițializez TinyLlama: {model_id}")
            
            # Detectează device-ul
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🖥️  Device: {self.device}")
            
            if self.device == "cuda":
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info(f"🎮 GPU: {gpu_name} ({total_memory:.1f}GB)")
                except:
                    logger.info("🎮 GPU detectat dar nu pot obține detalii")
            else:
                logger.info("🖥️  Rulează pe CPU - perfect pentru TinyLlama!")
            
            # Login HuggingFace dacă e necesar
            if hf_token:
                try:
                    login(token=hf_token)
                    logger.info("🔐 HuggingFace login reușit")
                except Exception as e:
                    logger.warning(f"HuggingFace login eșuat: {e}")
            
            # Creează cache dir
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
                logger.info(f"📁 Cache dir creat: {cache_dir}")
            
            # Încărcare tokenizer TinyLlama
            logger.info("📝 Încărcare tokenizer TinyLlama...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            # Setup pentru TinyLlama tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Adaugă chat template dacă nu există
            if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
                # Template simplu pentru TinyLlama
                self.tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
            
            logger.info("✅ Tokenizer TinyLlama încărcat")
            
            # Încărcare model TinyLlama
            logger.info("🧠 Încărcare model TinyLlama...")
            start_time = time.time()
            
            # Setări optimizate pentru TinyLlama
            model_kwargs = {
                "cache_dir": cache_dir,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float32,  # TinyLlama merge bine cu float32
            }
            
            # Device mapping pentru TinyLlama
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
                model_kwargs["torch_dtype"] = torch.float16  # Float16 pe GPU
            
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            
            load_time = time.time() - start_time
            logger.info(f"⏱️  Model încărcat în {load_time:.1f} secunde")
            
            # Info despre model
            self.model_name = model_id.split('/')[-1]
            self.is_tinyllama = "tinyllama" in model_id.lower()
            
            # Calculează parametrii
            try:
                total_params = sum(p.numel() for p in self.model.parameters())
                logger.info(f"🔢 Parametri: {total_params/1e9:.2f}B")
            except:
                logger.info("🔢 Nu pot calcula parametrii")
            
            # Test rapid de funcționare
            logger.info("🧪 Test funcționare...")
            if self._test_generation():
                self.initialized = True
                logger.info("✅ TinyLlama inițializat cu SUCCES!")
                logger.info("🚀 Gata pentru analiză rapidă!")
            else:
                raise Exception("Test de funcționare eșuat")
                
        except Exception as e:
            logger.error(f"❌ Eroare inițializare TinyLlama: {e}")
            self.initialized = False
            raise
    
    def _test_generation(self):
        """Test rapid pentru TinyLlama"""
        try:
            test_prompt = "Salut! Cum"
            
            inputs = self.tokenizer(test_prompt, return_tensors="pt")
            
            if self.device == "cuda" and next(self.model.parameters()).is_cuda:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"🧪 Test răspuns: '{response}'")
            return True
            
        except Exception as e:
            logger.error(f"🧪 Test eșuat: {e}")
            return False
    
    def generate_response(self, prompt, max_new_tokens=250, temperature=0.7, do_sample=True):
        """
        Generează răspuns optimizat pentru TinyLlama
        """
        if not self.initialized:
            raise RuntimeError("TinyLlama nu este inițializat!")
        
        try:
            # Import config pentru setări
            try:
                from app.config import Config
                
                max_new_tokens = getattr(Config, 'LLM_MAX_NEW_TOKENS', max_new_tokens)
                temperature = getattr(Config, 'LLM_DEFAULT_TEMPERATURE', temperature)
                do_sample = getattr(Config, 'LLM_DO_SAMPLE', do_sample)
            except ImportError:
                pass
            
            # Adaptează promptul pentru TinyLlama
            formatted_prompt = self._format_tinyllama_prompt(prompt)
            
            # Limitează lungimea pentru TinyLlama
            truncated_prompt = self._truncate_for_tinyllama(formatted_prompt)
            
            logger.info(f"🔄 TinyLlama generează răspuns...")
            logger.info(f"📏 Tokens noi: {max_new_tokens}, Temp: {temperature}")
            
            # Tokenizare optimizată pentru TinyLlama
            inputs = self.tokenizer(
                truncated_prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=1800  # Lasă spațiu pentru output
            )
            
            if self.device == "cuda" and next(self.model.parameters()).is_cuda:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            input_length = inputs['input_ids'].shape[1]
            logger.info(f"📏 Input tokens: {input_length}")
            
            # Parametri optimizați pentru TinyLlama
            generation_config = self._get_tinyllama_generation_config(
                max_new_tokens, temperature, do_sample
            )
            
            # Generare cu timing
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)
            
            generation_time = time.time() - start_time
            
            # Decodare și cleanup
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = self._extract_tinyllama_response(full_response, truncated_prompt)
            
            # Curăță răspunsul
            response = self._clean_tinyllama_response(response)
            
            # Statistici
            self.generation_stats["total_generations"] += 1
            self.generation_stats["total_time"] += generation_time
            
            logger.info(f"✅ Generat în {generation_time:.1f}s ({len(response)} chars)")
            
            if getattr(Config, 'WARN_IF_GENERATION_SLOW', 60) < generation_time:
                logger.warning(f"⚠️  Generare lentă: {generation_time:.1f}s")
            
            # Cleanup GPU memory
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Eroare generare TinyLlama: {e}")
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return f"Eroare la generare: {str(e)[:100]}..."
    
    def _format_tinyllama_prompt(self, prompt):
        """Formatează prompt pentru TinyLlama"""
        if self.is_tinyllama:
            # TinyLlama funcționează bine cu format simplu
            return f"<|user|>\n{prompt}\n<|assistant|>\n"
        return prompt
    
    def _truncate_for_tinyllama(self, prompt, max_length=1500):
        """Trunchiază prompt pentru TinyLlama (context limitat)"""
        if len(prompt) <= max_length:
            return prompt
        
        logger.warning(f"Prompt prea lung pentru TinyLlama ({len(prompt)} chars), trunchiez")
        
        # Păstrează începutul și sfârșitul
        start_part = prompt[:max_length//2]
        end_part = prompt[-(max_length//2):]
        
        return start_part + "\n[...]\n" + end_part
    
    def _get_tinyllama_generation_config(self, max_new_tokens, temperature, do_sample):
        """Configurație optimizată pentru TinyLlama (fără warning-uri)"""
        config = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        
        if do_sample:
            # Sampling mode pentru TinyLlama
            config.update({
                "do_sample": True,
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                # Nu folosim early_stopping cu greedy/sampling
            })
        else:
            # Greedy mode - fără early_stopping cu num_beams=1
            config.update({
                "do_sample": False,
                "repetition_penalty": 1.05,
                # Eliminat early_stopping pentru a evita warning-ul
            })
        
        return config
    
    def _extract_tinyllama_response(self, full_response, prompt):
        """Extrage răspunsul din output-ul TinyLlama"""
        # Elimină promptul original
        response = full_response.replace(prompt, "").strip()
        
        # Curăță marcajele TinyLlama
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        if "<|user|>" in response:
            response = response.split("<|user|>")[0].strip()
        
        return response
    
    def _clean_tinyllama_response(self, response):
        """Curăță răspunsul TinyLlama de artefacte"""
        if not response:
            return "Nu am putut genera un răspuns valid."
        
        # Elimină repetițiile comune la TinyLlama
        lines = response.split('\n')
        cleaned_lines = []
        prev_line = ""
        
        for line in lines:
            line = line.strip()
            if line and line != prev_line:  # Elimină liniile duplicate consecutive
                cleaned_lines.append(line)
                prev_line = line
        
        response = '\n'.join(cleaned_lines)
        
        # Pentru JSON, păstrează doar JSON-ul valid
        json_match = re.search(r'(\{[^{}]*"factuality_score"[^{}]*\})', response, re.IGNORECASE | re.DOTALL)
        if json_match:
            # Dacă găsim JSON, returnează doar JSON-ul + completează dacă e incomplet
            json_part = json_match.group(1)
            if not json_part.endswith('}'):
                json_part += '}'
            return json_part
        
        # Limitează lungimea dacă nu e JSON
        if len(response) > 500:
            response = response[:500] + "..."
        
        return response.strip()
    
    def get_model_info(self):
        """Info despre TinyLlama"""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        avg_time = 0
        if self.generation_stats["total_generations"] > 0:
            avg_time = self.generation_stats["total_time"] / self.generation_stats["total_generations"]
        
        return {
            "status": "initialized",
            "model_name": self.model_name,
            "device": self.device,
            "is_tinyllama": self.is_tinyllama,
            "total_generations": self.generation_stats["total_generations"],
            "average_time": f"{avg_time:.1f}s",
            "memory_efficient": True
        }