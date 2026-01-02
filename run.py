import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.python.bench import InferenceEngine, run_local_benchmark

def main():
    print("GLM-4.7-FP8 Inference Engine")
    print("=" * 50)
    print("This engine is designed to run on Modal.com with 8x B200 GPUs")
    print()
    print("To deploy to Modal:")
    print("  modal deploy src/python/modal_app.py")
    print()
    print("To run benchmark on Modal:")
    print("  modal run src/python/modal_app.py")
    print()
    print("Running local scheduler test...")
    print()
    
    engine = InferenceEngine(
        model_dir="./model",
        max_batch=4,
        max_seq=512,
        num_gpus=1
    )
    
    prompt = "Hello world"
    output = engine.generate(
        prompt=prompt,
        max_new_tokens=16,
        temperature=0.8,
        top_p=0.95,
        rep_penalty=1.1,
        stop_sequences=[]
    )
    
    print(f"Prompt: {prompt}")
    print(f"Output: {output}")
    print()
    print("Local test completed")
    
    engine.cleanup()

if __name__ == "__main__":
    main()
