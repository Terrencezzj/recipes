# Command-R-08-2024 and Command-R-plus-08-2024 Usage Guide

This guide describes how to run Command-R-08-2024 and Command-R-plus-08-2024 with BF16 and FP8. 

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Running Command-R-08-2024 and Command-R-plus-08-2024 with BF16

```bash

# Start server with BF16 model on 2 H100_80GB GPUs for R.
vllm serve CohereLabs/c4ai-command-r-08-2024 \
     --tensor-parallel-size 2 \
     --enable-chunked-prefill

# Start server with BF16 model on 4 H100_80GB GPUs for Rplus.
vllm serve CohereLabs/c4ai-command-r-plus-08-2024 \
     --tensor-parallel-size 4 \
     --enable-chunked-prefill
```

* You can set `--max-model-len` to preserve memory. `--max-model-len=65536` is usually good for most scenarios and max is 128k.
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency. `--max-num-batched-tokens=4096` is usually good for with chunked prefill enabled.
* vLLM conservatively use 90% of GPU memory, you can set `--gpu-memory-utilization=0.95` to maximize KVCache.

## Benchmarking

For benchmarking, you need to disable prefix caching by adding `--no-enable-prefix-caching` to the server command.

Once the server is running, open another terminal and run the benchmark client:

### BF16 Benchmark

```bash
# Prompt-heavy benchmark (10k/1k)
vllm bench serve \
  --model CohereLabs/c4ai-command-r-plus-08-2024 \
  --dataset-name random \
  --random-input-len 10000 \
  --random-output-len 1000 \
  --num-prompts 16
```

* Test different batch sizes by changing `--num-prompts`, e.g., 1, 16, 32, 64, 128, 256, 512

### Expected Output

##### Command-R-plus-08-2024

```shell
============ Serving Benchmark Result ============
Successful requests:                     16        
Benchmark duration (s):                  43.67     
Total input tokens:                      159984    
Total generated tokens:                  16000     
Request throughput (req/s):              0.37      
Output token throughput (tok/s):         366.35    
Total Token throughput (tok/s):          4029.47   
---------------Time to First Token----------------
Mean TTFT (ms):                          9143.33   
Median TTFT (ms):                        9053.47   
P99 TTFT (ms):                           16793.61  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          34.11     
Median TPOT (ms):                        34.20     
P99 TPOT (ms):                           41.38     
---------------Inter-token Latency----------------
Mean ITL (ms):                           34.11     
Median ITL (ms):                         26.82     
P99 ITL (ms):                            429.19    
==================================================
```

##### Command-R-08-2024

```shell
============ Serving Benchmark Result ============
Successful requests:                     16        
Benchmark duration (s):                  28.05     
Total input tokens:                      159984    
Total generated tokens:                  16000     
Request throughput (req/s):              0.57      
Output token throughput (tok/s):         570.46    
Total Token throughput (tok/s):          6274.44   
---------------Time to First Token----------------
Mean TTFT (ms):                          5348.32   
Median TTFT (ms):                        5291.35   
P99 TTFT (ms):                           9841.45   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          22.43     
Median TPOT (ms):                        22.49     
P99 TPOT (ms):                           26.65     
---------------Inter-token Latency----------------
Mean ITL (ms):                           22.43     
Median ITL (ms):                         18.19     
P99 ITL (ms):                            252.24    
==================================================
```


## Running Command-R-08-2024 and Command-R-plus-08-2024 with FP8
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

# Start server with FP8 model on 1 GPUs for R.
vllm serve path-to-CohereLabs/c4ai-command-r-08-2024-fp8 \
     --tensor-parallel-size 1 \
     --enable-chunked-prefill
     --max-num-batched-tokens 4096
     --max-model-len 128000
     --gpu-memory-utilization 0.95
     --quantization=compressed-tensors


# Start server with FP8 model on 2 GPUs for Rplus.
vllm serve path-to-CohereLabs/c4ai-command-r-plus-08-2024-fp8 \
     --tensor-parallel-size 2 \
     --enable-chunked-prefill
     --max-num-batched-tokens 4096
     --max-model-len 128000
     --gpu-memory-utilization 0.95
     --quantization=compressed-tensors
```
### Expected Output

##### Command-R-plus-08-2024 FP8

```shell
============ Serving Benchmark Result ============
Successful requests:                     16        
Benchmark duration (s):                  177.16    
Total input tokens:                      159984    
Total generated tokens:                  16000     
Request throughput (req/s):              0.09      
Output token throughput (tok/s):         90.31     
Total Token throughput (tok/s):          993.37    
---------------Time to First Token----------------
Mean TTFT (ms):                          11099.94  
Median TTFT (ms):                        10988.34  
P99 TTFT (ms):                           20388.51  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          151.16    
Median TPOT (ms):                        151.43    
P99 TPOT (ms):                           158.02    
---------------Inter-token Latency----------------
Mean ITL (ms):                           151.16    
Median ITL (ms):                         31.40     
P99 ITL (ms):                            553.65    
==================================================
```

##### Command-R-08-2024 FP8

```shell
============ Serving Benchmark Result ============
Successful requests:                     16        
Benchmark duration (s):                  34.60     
Total input tokens:                      159984    
Total generated tokens:                  16000     
Request throughput (req/s):              0.46      
Output token throughput (tok/s):         462.43    
Total Token throughput (tok/s):          5086.30   
---------------Time to First Token----------------
Mean TTFT (ms):                          6192.84   
Median TTFT (ms):                        6118.42   
P99 TTFT (ms):                           11424.11  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          28.10     
Median TPOT (ms):                        28.18     
P99 TPOT (ms):                           32.98     
---------------Inter-token Latency----------------
Mean ITL (ms):                           28.10     
Median ITL (ms):                         23.24     
P99 ITL (ms):                            292.51    
==================================================
```