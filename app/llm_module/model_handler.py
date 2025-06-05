import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import snapshot_download, login, HfFolder
import logging

logger = logging.getLogger(__name__)

class ModelHandler:
    """
    Gestionează modelul LLM upgraded - folosește pattern Singleton și suportă modele mai mari
    UPGRADE: Suport pentru Llama 3.1 8B cu quantization 4-bit pentru eficiență RAM
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelHandler, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.tokenizer = None
            cls._instance.device = None
            cls._instance.initialized = False
            cls._instance.max_context_length = 8192  # Llama 3.1 suportă mult mai mult context
            cls._instance.model_name = None
            cls._instance.quantized = False
        return cls._instance
    
    def initialize(self, 
                  model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",  # Upgrade la Llama 3.1
                  cache_dir="./model_cache", 
                  use_4bit=True,  # Activează quantization by default
                  hf_token=None):
        """
        Inițializează modelul upgraded - suportă modele mari cu quantization
        
        Args:
            model_id: ID-ul modelului (default: Llama 3.1 8B)
            cache_dir: Directorul de cache
            use_4bit: Folosește quantization 4-bit pentru RAM efficiency
            hf_token: Token Hugging Face (necesar pentru unele modele)
        """
        if self.initialized:
            logger.info(f"Modelul {self.model_name} este deja inițializat")
            return
        
        try:
            logger.info(f"🚀 Începere inițializare model UPGRADED: {model_id}")
            
            # Detectează device-ul disponibil
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"🖥️  Device detectat: {self.device}")
            
            # Afișează info despre GPU dacă e disponibil
            if self.device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"🎮 GPU: {gpu_name}")
                logger.info(f"💾 Memorie GPU totală: {total_memory:.1f} GB")
                
                # Verifică dacă avem destulă memorie pentru modelul mare
                if total_memory < 6 and not use_4bit:
                    logger.warning("⚠️  GPU cu <6GB RAM - activez quantization automată")
                    use_4bit = True
            else:
                logger.info("⚠️  Rulând pe CPU - va fi mai lent dar funcționează")
            
            # Login la Hugging Face dacă e necesar
            if hf_token:
                login(token=hf_token)
                logger.info("🔐 Autentificare Hugging Face completă")
            
            # Verifică și creează directorul de cache
            if not os.path.exists(cache_dir):
                logger.info(f"📁 Creez directorul de cache: {cache_dir}")
                os.makedirs(cache_dir, exist_ok=True)
            
            # Configurează quantization dacă e activată
            quantization_config = None
            if use_4bit:
                logger.info("⚡ Configurez quantization 4-bit pentru economie de RAM")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                self.quantized = True
            
            # Încărcare tokenizer
            logger.info("📝 Încărcare tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            
            # Setup padding token dacă nu există
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("🔧 Setat padding token")
            
            # Încărcare model cu configurația optimă
            logger.info(f"🧠 Încărcare model {model_id}...")
            if use_4bit:
                logger.info("   ⚡ Folosind quantization 4-bit - va dura ~2-3 minute")
            else:
                logger.info("   ⚡ Încărcare model complet - va dura ~5-10 minute")
            
            torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                quantization_config=quantization_config,
                device_map="auto" if self.device == "cuda" else None,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Informații post-încărcare
            self.model_name = model_id.split('/')[-1]
            
            if self.device == "cuda":
                try:
                    torch.cuda.empty_cache()  # Curățăm cache-ul GPU
                    allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    cached = torch.cuda.memory_reserved(0) / (1024**3)
                    logger.info(f"💾 Memorie GPU folosită: {allocated:.1f}GB (cached: {cached:.1f}GB)")
                except:
                    logger.info("💾 Nu pot afișa info despre memoria GPU")
            
            # Estimează parametrii efectivi
            param_count = sum(p.numel() for p in self.model.parameters()) / 1e9
            logger.info(f"🔢 Parametri model: ~{param_count:.1f}B")
            
            if self.quantized:
                logger.info("⚡ Quantization activă - modelul folosește ~50% mai puțin RAM")
            
            # Test rapid pentru a verifica funcționarea
            logger.info("🧪 Test rapid de funcționare...")
            test_result = self._quick_functionality_test()
            
            if test_result:
                self.initialized = True
                logger.info("✅ Model inițializat cu SUCCES!")
                logger.info(f"🎯 Context maxim: {self.max_context_length} tokens")
                logger.info(f"🚀 Gata pentru analiză avansată de factualitate!")
            else:
                raise Exception("Testul de funcționare a eșuat")
                
        except Exception as e:
            logger.error(f"❌ Eroare la inițializarea modelului: {str(e)}")
            self.initialized = False
            raise
    
    def _quick_functionality_test(self):
        """Test rapid pentru a verifica că modelul funcționează"""
        try:
            test_prompt = "Test: 2+2="
            inputs = self.tokenizer(test_prompt, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            logger.info(f"🧪 Test răspuns: '{response}'")
            return True
            
        except Exception as e:
            logger.error(f"🧪 Test eșuat: {e}")
            return False
    
    def _truncate_input(self, prompt, max_input_length=6000):  # Mărită limita pentru Llama 3.1
        """Trunchiază input-ul pentru a se încadra în contextul modelului"""
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        if len(tokens) <= max_input_length:
            return prompt
        
        logger.warning(f"⚠️  Prompt prea lung ({len(tokens)} tokens), trunchiez la {max_input_length}")
        
        # Pentru modele mai puternice, păstrăm mai mult context
        start_tokens = tokens[:max_input_length//2]
        end_tokens = tokens[-(max_input_length//2):]
        
        start_text = self.tokenizer.decode(start_tokens, skip_special_tokens=True)
        end_text = self.tokenizer.decode(end_tokens, skip_special_tokens=True)
        
        truncated_prompt = start_text + "\n[...text trunchiat pentru analiza optimă...]\n" + end_text
        return truncated_prompt
    
    def generate_response(self, prompt, max_new_tokens=512, temperature=0.7, do_sample=True):
        """
        Generează răspuns optimizat pentru Llama 3.1
        
        Args:
            prompt: Promptul de input
            max_new_tokens: Numărul maxim de tokeni noi
            temperature: Temperatura pentru sampling
            do_sample: Dacă să folosească sampling
        """
        if not self.initialized:
            raise RuntimeError("⚠️  Modelul nu este inițializat! Apelează initialize() mai întâi.")
        
        try:
            # Optimizează promptul pentru Llama 3.1 (folosește format chat)
            if "meta-llama/Meta-Llama-3" in self.model_name:
                formatted_prompt = self._format_llama_prompt(prompt)
            else:
                formatted_prompt = prompt
            
            # Trunchiază dacă e necesar
            truncated_prompt = self._truncate_input(formatted_prompt, max_input_length=6000)
            
            logger.info(f"🔄 Generez răspuns cu {self.model_name} (primele 50 char): '{truncated_prompt[:50]}...'")
            
            # Tokenizare cu handling îmbunătățit
            inputs = self.tokenizer(
                truncated_prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=self.max_context_length - max_new_tokens
            )
            
            # Mută pe device-ul corect
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            input_length = inputs['input_ids'].shape[1]
            logger.info(f"📏 Lungime input: {input_length} tokens")
            
            # Generare cu parametri optimizați pentru factualitate
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": 0.9,  # Nucleus sampling pentru răspunsuri mai focalizate
                "repetition_penalty": 1.1,  # Evită repetițiile
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Pentru analiză factualitate, folosim parametri mai conservatori
            if temperature < 0.5:  # Probabil factuality analysis
                generation_config.update({
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "repetition_penalty": 1.05
                })
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)
            
            # Decodează doar partea nouă
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extrage doar răspunsul nou (elimină promptul)
            if "meta-llama/Meta-Llama-3" in self.model_name:
                response = self._extract_llama_response(full_response, formatted_prompt)
            else:
                response = full_response.replace(truncated_prompt, "").strip()
            
            if not response:
                response = full_response  # Fallback
            
            logger.info(f"✅ Răspuns generat: {len(response)} caractere")
            logger.info(f"🔍 Preview răspuns: {response[:100]}...")
            
            # Curăță memoria GPU dacă e necesar
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Eroare la generarea răspunsului: {str(e)}")
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return f"Eroare la procesarea textului cu {self.model_name}: {str(e)[:100]}... Te rog încearcă cu un text mai scurt."
    
    def _format_llama_prompt(self, prompt):
        """Formatează promptul pentru Llama 3.1 folosind chat template"""
        messages = [
            {
                "role": "system", 
                "content": "Ești un asistent AI expert în verificarea factualității și analiza critică a textelor. Răspunde întotdeauna în JSON când este solicitat."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        # Folosește chat template dacă e disponibil
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            # Fallback manual pentru Llama format
            formatted = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            formatted += messages[0]["content"]
            formatted += "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            formatted += messages[1]["content"]
            formatted += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            return formatted
    
    def _extract_llama_response(self, full_response, original_prompt):
        """Extrage răspunsul din output-ul Llama 3.1"""
        # Încearcă să găsească răspunsul după promptul formatat
        if "<|start_header_id|>assistant<|end_header_id|>" in full_response:
            parts = full_response.split("<|start_header_id|>assistant<|end_header_id|>")
            if len(parts) > 1:
                response = parts[-1].strip()
                # Elimină token-ii de sfârșitul răspunsului
                response = response.replace("<|eot_id|>", "").strip()
                return response
        
        # Fallback: elimină promptul original
        return full_response.replace(original_prompt, "").strip()
    
    def get_model_info(self):
        """Returnează informații despre modelul încărcat"""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "initialized",
            "model_name": self.model_name,
            "device": self.device,
            "quantized": self.quantized,
            "max_context": self.max_context_length,
            "parameters": f"~{sum(p.numel() for p in self.model.parameters()) / 1e9:.1f}B"
        }