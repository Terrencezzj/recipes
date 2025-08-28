# Command-A, Command-A-Reasoning and Command-A-Vision Usage Guide

This guide describes how to run Command-A, Command-A-Reasoning and Command-A-Vision with BF16 and FP8. 

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Running Command-A with BF16

```bash

# Start server with BF16 model on 4 H100_80GB GPUs for Command-A.
vllm serve CohereLabs/c4ai-command-a-03-2025 \
     --tensor-parallel-size 4 \
     --enable-chunked-prefill
```

* You can set `--max-model-len` to preserve memory. `--max-model-len=65536` is usually good for most scenarios and max is 256k.
     Note: The model supports a context length of 256K but it is configured in Hugging Face for 128K. This value can be updated in the config.json if needed.
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency. `--max-num-batched-tokens=4096` is usually good for with chunked prefill enabled.
* vLLM conservatively use 90% of GPU memory, you can set `--gpu-memory-utilization=0.95` to maximize KVCache.

## Benchmarking

For benchmarking, you need to disable prefix caching by adding `--no-enable-prefix-caching` to the server command.

Once the server is running, open another terminal and run the benchmark client:

### BF16 Benchmark

```bash
# Prompt-heavy benchmark (10k/1k)
vllm bench serve \
  --model CohereLabs/c4ai-command-a-03-2025 \
  --dataset-name random \
  --random-input-len 10000 \
  --random-output-len 1000 \
  --num-prompts 16
```

* Test different batch sizes by changing `--num-prompts`, e.g., 1, 16, 32, 64, 128, 256, 512

### Expected Output

##### Command-A

```shell
============ Serving Benchmark Result ============
Successful requests:                     16        
Benchmark duration (s):                  42.29     
Total input tokens:                      159984    
Total generated tokens:                  16000     
Request throughput (req/s):              0.38      
Output token throughput (tok/s):         378.36    
Total Token throughput (tok/s):          4161.55   
---------------Time to First Token----------------
Mean TTFT (ms):                          8136.29   
Median TTFT (ms):                        8241.76   
P99 TTFT (ms):                           15785.82  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          33.72     
Median TPOT (ms):                        33.63     
P99 TPOT (ms):                           41.10     
---------------Inter-token Latency----------------
Mean ITL (ms):                           33.72     
Median ITL (ms):                         26.41     
P99 ITL (ms):                            430.96    
==================================================
```

## Running Command-A and Command-A-Reasoning with FP8
To get FP8 checkpoint,  `llmcompressor` is required. 
You can use the following script to convert FP8 checkpoint.
#### Example script

```python
def quantize_to_fp8(source_dir, output_dir):
    from llmcompressor.transformers import oneshot
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from llmcompressor.modifiers.quantization import QuantizationModifier

    os.makedirs(output_dir, exist_ok = True)
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    model = AutoModelForCausalLM.from_pretrained(source_dir, torch_dtype="auto")

    quant_recipe = QuantizationModifier(targets = "Linear",
                                        scheme = "FP8_DYNAMIC",
                                        ignore = ['re:.*lm_head'],
                                        kv_cache_scheme = None)

    # Apply the quantization algorithm.
    oneshot(
        model=model,
        recipe=quant_recipe,
        tokenizer=tokenizer,
    )
    model.save_pretrained(output_dir, save_compressed=True, skip_compression_stats=True)
    tokenizer.save_pretrained(output_dir)
```

## Running Command-R-08-2024 and Command-R-plus-08-2024 with FP8

```bash

# Start server with FP8 model on 2 H100_80GB GPUs for Command-A.
vllm serve path-to-CohereLabs/c4ai-command-a-03-2025-fp8 \
     --tensor-parallel-size 2 \
     --enable-chunked-prefill
     --max-num-batched-tokens 4096
     --max-model-len 256000
     --gpu-memory-utilization 0.95
     --quantization=compressed-tensors

```
### Expected Output

##### Command-A FP8

```shell
============ Serving Benchmark Result ============
Successful requests:                     16        
Benchmark duration (s):                  49.39     
Total input tokens:                      159984    
Total generated tokens:                  16000     
Request throughput (req/s):              0.32      
Output token throughput (tok/s):         323.94    
Total Token throughput (tok/s):          3563.07   
---------------Time to First Token----------------
Mean TTFT (ms):                          10372.79  
Median TTFT (ms):                        10502.55  
P99 TTFT (ms):                           20051.97  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          38.56     
Median TPOT (ms):                        38.45     
P99 TPOT (ms):                           48.06     
---------------Inter-token Latency----------------
Mean ITL (ms):                           38.56     
Median ITL (ms):                         29.26     
P99 ITL (ms):                            538.39    
==================================================
```
