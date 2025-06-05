# test_minimal_llm.py

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configurează logging-ul
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tiny_model():
    """Testează un model foarte mic pentru a verifica funcționalitatea."""
    try:
        logger.info("Încărcare model mic de test...")
        
        # Folosește un model foarte mic pentru test
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        logger.info(f"Încărcare tokenizer pentru {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        logger.info(f"Încărcare model {model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Prompt simplu pentru test
        prompt = "Ce este un model de limbaj?"
        
        logger.info(f"Generare răspuns pentru prompt: '{prompt}'")
        inputs = tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                temperature=0.7,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info("Test reușit!")
        logger.info(f"Răspuns generat: {response}")
        
        return True
    except Exception as e:
        logger.error(f"Eroare la testarea modelului: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testare funcționalitate modele de limbaj...\n")
    print("Acest test va încerca să încarce un model mic de limbaj.")
    print("Va dura câteva minute și va descărca aproximativ 600MB de date.")
    print("Continuă? (da/nu)")
    
    response = input().strip().lower()
    if response != "da":
        print("Test anulat.")
        exit()
    
    success = test_tiny_model()
    
    if success:
        print("\n✓ Testul a reușit! Sistemul tău poate rula modele de limbaj.")
        print("Poți continua cu implementarea completă a modelului Mistral.")
    else:
        print("\n✗ Testul a eșuat. Verifică erorile pentru mai multe detalii.")