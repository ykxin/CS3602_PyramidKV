import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import time
import torch
import math
import numpy as np
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from datasets import load_dataset
from pyramidkv.monkeypatch import replace_gptneox
import gc

# Set environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_ppl(model, tokenizer, text, stride=512, device="cuda"):
    """
    Evaluates PPL using a Chunked Context-Target approach to properly test PyramidKV compression.
    Instead of a sliding window that might reset compression state, we split the text into chunks.
    Each chunk is split into Context (Prompt) and Target.
    Context is processed to generate compressed KV cache.
    Target is evaluated using that compressed cache.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    total_len = input_ids.size(1)
    
    # Configuration for Chunked Evaluation
    # We use a chunk size of 2048 (standard for Pythia/GPT-NeoX)
    # Context: 1536 tokens (75%), Target: 512 tokens (25%)
    seq_len = 2048
    split_idx = int(seq_len * 0.75)
    
    nlls = []
    
    print(f"Evaluating PPL with Chunk Size {seq_len}, Context Split {split_idx}...")
    
    # Iterate through text in non-overlapping chunks (or with stride if needed)
    # Using non-overlapping chunks for efficiency and independence
    count = 0
    for i in range(0, total_len, seq_len):
        chunk_ids = input_ids[:, i : i + seq_len].to(device)
        
        # Skip if chunk is too short to have a meaningful context+target split
        if chunk_ids.size(1) < seq_len:
            break
            
        context_ids = chunk_ids[:, :split_idx]
        target_ids = chunk_ids[:, split_idx:]
        
        # 1. Process Context (Trigger Compression)
        # This simulates reading a long document/prompt
        with torch.no_grad():
            outputs = model(context_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            
        # 2. Process Target (Evaluate with Compressed Context)
        # We compute the likelihood of target_ids given context
        # We need correct position_ids for RoPE
        past_length = context_ids.size(1)
        target_length = target_ids.size(1)
        position_ids = torch.arange(past_length, past_length + target_length, dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            # Pass labels=target_ids to compute loss. 
            # Note: The model shifts labels internally, so it predicts target_ids[1:] given target_ids[:-1] and context.
            # The first token of target_ids is used as input, but its prediction is not part of the loss (usually).
            outputs_ppl = model(
                target_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
                labels=target_ids
            )
            neg_log_likelihood = outputs_ppl.loss
            
        nlls.append(neg_log_likelihood)
        count += 1
        
        # Limit evaluation to save time if text is huge (e.g. max 50 chunks)
        if count >= 50:
            break

    if not nlls:
        return float('nan')
    
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def benchmark_speed(model, tokenizer, prompt, new_tokens=100, device="cuda"):
    """
    Benchmarks TTFT, TPOT, and Memory usage with correct cache reuse.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    print(f"Prompt length (tokens): {input_ids.shape[1]}")
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated()
    
    # Warmup
    print("Warmup...")
    _ = model.generate(input_ids, max_new_tokens=10, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    
    # Clear cache from warmup
    torch.cuda.empty_cache()
    
    # 1. TTFT (Time To First Token) - Prefill
    print("Measuring TTFT...")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    
    torch.cuda.synchronize()
    ttft = time.time() - start_time
    
    # Memory after Prefill (contains KV cache for the prompt)
    mem_after_prefill = torch.cuda.memory_allocated()
    kv_cache_mem_est = (mem_after_prefill - start_mem) / (1024**3) # GB (Approx)
    print(f"Memory after prefill: {mem_after_prefill / 1024**3:.4f} GB")
    
    # 2. TPOT (Time Per Output Token)
    print(f"Generating {new_tokens} tokens for TPOT measurement...")
    
    past_key_values = outputs.past_key_values
    last_token = input_ids[:, -1:]
    
    # We need to correctly handle position_ids if we generate manually, 
    # but model.generate handles it if we pass past_key_values.
    
    start_time = time.time()
    
    # Using past_key_values avoids re-computation of prefill
    gen_output = model.generate(
        last_token, 
        past_key_values=past_key_values,
        max_new_tokens=new_tokens, 
        use_cache=True, 
        pad_token_id=tokenizer.eos_token_id
    )
    
    torch.cuda.synchronize()
    total_gen_time = time.time() - start_time
    
    # gen_output contains [last_token, new_tokens...]
    gen_tokens = gen_output.shape[1] - 1 
    
    tpot = total_gen_time / gen_tokens if gen_tokens > 0 else 0
    throughput = gen_tokens / total_gen_time
    
    # FLOPS estimation
    num_params = sum(p.numel() for p in model.parameters())
    total_flops = 2 * num_params * gen_tokens
    flops_achieved = total_flops / total_gen_time
    
    max_memory = torch.cuda.max_memory_allocated() / (1024 ** 3) # GB
    
    return {
        "TTFT (s)": ttft,
        "TPOT (s)": tpot,
        "Throughput (tokens/s)": throughput,
        "FLOPS (TFLOPS)": flops_achieved / 1e12,
        "Max Memory (GB)": max_memory,
        "KV Cache Mem (GB)": kv_cache_mem_est
    }

def load_data(tokenizer):
    print("\nLoading datasets...")
    
    # WikiText
    try:
        wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        wikitext_text = "\n".join(wikitext['text'])
        print(f"WikiText loaded (len: {len(wikitext_text)})")
    except Exception as e:
        print(f"Failed to load WikiText: {e}")
        wikitext_text = "This is a fallback text for WikiText. " * 1000

    # PG-19
    try:
        # Try loading a small subset or streaming
        pg19 = load_dataset("deepmind/pg19", split="validation", streaming=True, trust_remote_code=True)
        pg19_iter = iter(pg19)
        pg19_sample = next(pg19_iter)['text']
        print(f"PG-19 sample loaded (len: {len(pg19_sample)})")
    except Exception as e:
        print(f"Failed to load PG-19: {e}")
        print("Falling back to WikiText-103 as proxy...")
        try:
            wt103 = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", streaming=True)
            wt103_iter = iter(wt103)
            # Skip first few articles to avoid overlap with WikiText-2 (which is a subset)
            for _ in range(10): 
                try:
                    next(wt103_iter)
                except StopIteration:
                    break
            
            pg19_sample = ""
            # Accumulate enough text
            while len(pg19_sample) < 100000:
                try:
                    pg19_sample += next(wt103_iter)['text'] + "\n"
                except StopIteration:
                    break
            print(f"WikiText-103 loaded as proxy (len: {len(pg19_sample)})")
        except Exception as e2:
            print(f"Failed to load WikiText-103: {e2}")
            # Create a long text from WikiText (repeat 5 times ~ 1M chars)
            # Offset to avoid identical evaluation if both look at start
            print("Using extended WikiText-2 with offset...")
            pg19_sample = (wikitext_text * 5)[10000:]
        
    return wikitext_text, pg19_sample

def run_benchmark(model_name, method="baseline"):
    print(f"\n{'='*20} Running Benchmark: {method} {'='*20}")
    
    # Setup Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Apply Patch if needed
    if method == "pyramidkv":
        replace_gptneox("pyramidkv")
    else:
        # Restore original forward (naive way: rely on clean process or reload, 
        # but here we rely on manual restoration if running in same process)
        # However, for simplicity in this script, we assume 'baseline' runs first 
        # and we save the original forward before patching.
        pass

    print("Loading model...")
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    # Inject layer_idx into attention modules for PyramidKV (GPT-NeoX doesn't store it by default)
    if hasattr(model, "gpt_neox"):
        for i, layer in enumerate(model.gpt_neox.layers):
            layer.attention.layer_idx = i
    
    # Configure PyramidKV if active
    if method == "pyramidkv":
        model.config.window_size = 32
        model.config.max_capacity_prompt = 512 
        # model.config.num_hidden_layers is available
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Data
    wikitext_text, pg19_sample = load_data(tokenizer)
    
    results = {}
    
    # 1. PPL on WikiText (subset)
    print("\nRunning PPL Test on WikiText (subset)")
    ppl_wiki = get_ppl(model, tokenizer, wikitext_text[:20000], stride=512, device=device)
    print(f"WikiText PPL: {ppl_wiki:.4f}")
    results["WikiText PPL"] = ppl_wiki

    # 2. PPL on PG-19 (subset/proxy)
    print("\nRunning PPL Test on PG-19 (subset)")
    ppl_pg19 = get_ppl(model, tokenizer, pg19_sample[:20000], stride=512, device=device)
    print(f"PG-19 PPL: {ppl_pg19:.4f}")
    results["PG-19 PPL"] = ppl_pg19

    # 3. Acceleration Test
    print("\nRunning Acceleration Test")
    # Prompt length: ~2500 tokens (fits in 2.8B context usually, or slightly over)
    # 10000 chars is roughly 2500-3000 tokens
    prompt_text = pg19_sample[:10000] 
    metrics = benchmark_speed(model, tokenizer, prompt_text, new_tokens=128, device=device)
    
    results.update(metrics)
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return results

def main():
    model_name = "EleutherAI/pythia-2.8b"
    
    # Save original forward
    original_forward = GPTNeoXAttention.forward
    
    # Run Baseline
    baseline_results = run_benchmark(model_name, method="baseline")
    # baseline_results = {}
    
    # Run PyramidKV
    # Note: run_benchmark will call replace_gptneox which patches the class
    pyramid_results = run_benchmark(model_name, method="pyramidkv")
    
    # Comparison
    print(f"\n{'='*20} Comparison Results {'='*20}")
    print(f"{'Metric':<25} | {'Baseline':<15} | {'PyramidKV':<15} | {'Improvement'}")
    print("-" * 75)
    
    metrics = ["WikiText PPL", "PG-19 PPL", "TTFT (s)", "TPOT (s)", "Throughput (tokens/s)", "FLOPS (TFLOPS)", "Max Memory (GB)", "KV Cache Mem (GB)"]
    
    for metric in metrics:
        base = baseline_results.get(metric, float('nan'))
        pyra = pyramid_results.get(metric, float('nan'))
        
        if metric in ["Throughput (tokens/s)", "FLOPS (TFLOPS)"]:
            imp = (pyra - base) / base * 100 # Higher is better
            imp_str = f"{imp:+.2f}%"
        elif metric in ["WikiText PPL", "PG-19 PPL"]:
            imp = (pyra - base) / base * 100 # Lower is usually better (but change is what we measure)
            imp_str = f"{imp:+.2f}%"
        else:
            imp = (base - pyra) / base * 100 # Lower is better (Time, Memory)
            imp_str = f"{imp:+.2f}% (Reduction)"
            
        print(f"{metric:<25} | {base:<15.4f} | {pyra:<15.4f} | {imp_str}")
        
    print("=" * 75)

if __name__ == "__main__":
    main()
