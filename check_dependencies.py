# check_dependencies.py

def check_dependency(name):
    try:
        module = __import__(name)
        print(f"✓ {name} este instalat (versiunea: {getattr(module, '__version__', 'necunoscută')})")
        return True
    except ImportError:
        print(f"✗ {name} NU este instalat")
        return False

def main():
    print("Verificare dependențe pentru modelul LLM:\n")
    
    # Verifică dependențele principale
    torch_ok = check_dependency('torch')
    transformers_ok = check_dependency('transformers')
    accelerate_ok = check_dependency('accelerate')
    bitsandbytes_ok = check_dependency('bitsandbytes')
    sentencepiece_ok = check_dependency('sentencepiece')
    huggingface_ok = check_dependency('huggingface_hub')
    
    # Verifică dacă toate dependențele sunt instalate
    all_ok = all([torch_ok, transformers_ok, huggingface_ok])
    
    print("\nRezultat verificare:")
    if all_ok:
        print("✓ Toate dependențele principale sunt instalate!")
        
        # Dacă PyTorch este instalat, verifică disponibilitatea GPU
        if torch_ok:
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"✓ CUDA este disponibil - GPU detectat: {torch.cuda.get_device_name(0)}")
                    print(f"  Memorie GPU: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
                else:
                    print("✗ CUDA nu este disponibil - LLM va rula pe CPU (semnificativ mai lent)")
            except:
                print("✗ Nu s-a putut verifica disponibilitatea GPU")
    else:
        print("✗ Unele dependențe lipsesc.")
        print("\nPentru a instala dependențele lipsă, rulează:")
        
        if not torch_ok:
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
        
        missing = []
        if not transformers_ok: missing.append("transformers")
        if not accelerate_ok: missing.append("accelerate")
        if not sentencepiece_ok: missing.append("sentencepiece")
        if not huggingface_ok: missing.append("huggingface_hub")
        if not bitsandbytes_ok: missing.append("bitsandbytes")
        
        if missing:
            print(f"pip install {' '.join(missing)}")

if __name__ == "__main__":
    main()