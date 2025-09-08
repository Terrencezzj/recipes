# Command-A-Vision Usage Guide

This guide describes how to run Command-A-Vision with BF16 and FP8. 

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Running Command-A, Command-A-Reasoning, Command-A-Vision with BF16

```bash
# Start server with BF16 model on 4 H100_80GB GPUs for Command-A-Vision.
vllm serve CohereLabs/command-a-vision-07-2025 \
     --tensor-parallel-size 4 \
     --enable-chunked-prefill
```

* You can set `--max-model-len` to preserve memory. `--max-model-len=65536` is usually good for most scenarios.
     * Command-A-Vision support a context length of 128K but it is configured in Hugging Face for 32K. This value can be updated in the config.json if needed. 
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency. `--max-num-batched-tokens=4096` is usually good for with chunked prefill enabled.
* vLLM conservatively use 90% of GPU memory, you can set `--gpu-memory-utilization=0.95` to maximize KVCache.

### Example of Command-A-Vision
```shell
# Online Chat with server
python3 examples/online_serving/openai_chat_completion_client_for_multimodal.py --chat-type multi-image
# Output
Chat completion output: The images depict a mallard duck and a lion. The mallard duck is a common wild duck known for its iridescent green head, yellow bill, and brown body. The lion is a large carnivorous mammal, often referred to as the "king of the jungle," characterized by its golden-brown coat and,

# Offline inference
python3 examples/offline_inference/vision_language.py --model-type command_a_vision
# Output
The image captures a stunning view of the Tokyo Tower, a prominent landmark in Tokyo, Japan, framed by the delicate pink blossoms of cherry trees. The tower, with its white and orange structure, stands tall against a clear blue sky, creating a striking contrast. The cherry blossoms, in full bloom, dominate the foreground
```

## Convert Command-A-Vision to FP8
To get FP8 checkpoint,  `llmcompressor` is required. 
You can use the following script to convert FP8 checkpoint.
#### Example script

```python
import os
from llmcompressor.transformers import oneshot
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.modifiers.quantization import QuantizationModifier

def quantize_to_fp8(source_dir, output_dir):
    os.makedirs(output_dir, exist_ok = True)
    tokenizer = AutoTokenizer.from_pretrained(source_dir)
    model = AutoModelForCausalLM.from_pretrained(source_dir, dtype="auto")

    quant_recipe = QuantizationModifier(targets = "Linear",
                                        scheme = "FP8_DYNAMIC",
                                        ignore = ['re:.*lm_head', 're:multi_modal_projector.*', 're:vision_tower.*'],
                                        kv_cache_scheme = None)

    # Apply the quantization algorithm.
    oneshot(
        model=model,
        recipe=quant_recipe,
        tokenizer=tokenizer,
        tie_word_embeddings=True,
    )
    model.save_pretrained(output_dir, save_compressed=True, skip_compression_stats=True)
    tokenizer.save_pretrained(output_dir)
```

## Running Command-A-Vision with FP8

```bash
# Start server with FP8 model on 2 H100_80GB GPUs for Command-A-Vision.
vllm serve path-to-CohereLabs/command-a-vision-07-2025-fp8 \
     --tensor-parallel-size 2 \
     --enable-chunked-prefill
     --max-num-batched-tokens 4096
     --max-model-len 128000
     --gpu-memory-utilization 0.95
     --quantization=compressed-tensors

```

## Benchmarking

For benchmarking, you need to disable prefix caching by adding `--no-enable-prefix-caching` to the server command.

Once the server is running, open another terminal and run the benchmark client:

```bash
# Command-A-Vision
uv pip install vllm[bench]
vllm bench serve \
  --endpoint-type openai-chat \
  --backend openai-chat \
  --endpoint /v1/chat/completions \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --hf-split train \
  --model CohereLabs/command-a-vision-07-2025 \
  --num-prompts 16 \
  --hf-output-len 100
```

* Test different batch sizes by changing `--num-prompts`, e.g., 1, 16, 32, 64, 128, 256, 512

### Expected Output

##### Command-A-Vision
```shell
============ Serving Benchmark Result ============
Successful requests:                     16        
Benchmark duration (s):                  5.63      
Total input tokens:                      607       
Total generated tokens:                  1520      
Request throughput (req/s):              2.84      
Output token throughput (tok/s):         269.89    
Total Token throughput (tok/s):          377.67    
---------------Time to First Token----------------
Mean TTFT (ms):                          2158.94   
Median TTFT (ms):                        2291.42   
P99 TTFT (ms):                           3164.64   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          39.43     
Median TPOT (ms):                        33.45     
P99 TPOT (ms):                           107.46    
---------------Inter-token Latency----------------
Mean ITL (ms):                           34.92     
Median ITL (ms):                         24.85     
P99 ITL (ms):                            872.34    
==================================================
```
