# Command-A-Reasoning Usage Guide

This guide describes how to run Command-A-Reasoning with BF16 and FP8. 

## Installing vLLM

```bash
uv venv
source .venv/bin/activate
uv pip install -U vllm --torch-backend auto
```

## Running Command-A-Reasoning with BF16

```bash
# Start server with BF16 model on 4 H100_80GB GPUs for Command-A-Reasoning.
vllm serve CohereLabs/command-a-reasoning-08-2025 \
     --tensor-parallel-size 4 \
     --enable-chunked-prefill
```

* You can set `--max-model-len` to preserve memory. `--max-model-len=65536` is usually good for most scenarios.
     * Command-A-Reasoning support a context length of 256K but it is configured in Hugging Face for 128K. This value can be updated in the config.json if needed. 
* You can set `--max-num-batched-tokens` to balance throughput and latency, higher means higher throughput but higher latency. `--max-num-batched-tokens=4096` is usually good for with chunked prefill enabled.
* vLLM conservatively use 90% of GPU memory, you can set `--gpu-memory-utilization=0.95` to maximize KVCache.

### curl Example for Reasoning

You can run the following `curl` command:

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "CohereLabs/command-a-reasoning-08-2025",
        "messages": [
            {
                "role": "user",
                "content": "9.11 and 9.8, which is greater?"
            }
        ],
        "chat_template_kwargs": {
            "thinking": true
        }
    }'
```
#### Example Output
```markdown
<|START_THINKING|>Okay, so I need to figure out whether 9.11 is greater than 9.8 or if 9.8 is greater than 9.11. Let me think about how to approach this. Both numbers are decimals, and they both have the same whole number part, which is 9. When comparing decimals, if the whole number parts are equal, you look at the tenths place next. 

So, let's break it down. The first number is 9.11. The digit in the tenths place here is 1, and then there's an additional 1 in the hundredths place. The second number is 9.8. The tenths place here is 8, and since there's no digit after that, the hundredths place is 0. 

Since both numbers have the same whole number part, we compare the tenths place. The tenths place in 9.11 is 1, and in 9.8, it's 8. Wait, hold on, 8 is greater than 1, right? So does that mean that 9.8 is greater than 9.11? But let me make sure I'm not making a mistake here. Sometimes when there's an extra digit in the hundredths place, it's easy to get confused.

Let me visualize both numbers with the same number of decimal places to make it clearer. If I write 9.11 as is, it's 9.11. If I write 9.8 as 9.80, adding a zero in the hundredths place to match the two decimal places of 9.11. Now, comparing 9.11 and 9.80. 

Starting from the left, both have 9. Then tenths place: 1 vs. 8. Since 8 is greater than 1, even though 9.11 has an extra 1 in the hundredths place, the tenths place is more significant. So the 8 in the tenths place makes 9.80 larger than 9.11. Therefore, 9.8 is greater than 9.11. 

Alternatively, I can subtract the two numbers to check. Let's subtract 9.11 from 9.8. 9.8 minus 9.11. To do this, align the decimals:

9.80
-9.11
------
0.69

So 9.8 minus 9.11 equals 0.69, which is a positive number. That means 9.8 is indeed larger than 9.11 by 0.69. 

Another way to think about it is converting them to fractions. 9.11 can be written as 9 and 11/100, and 9.8 can be written as 9 and 80/100 (since 0.8 is 80/100). Comparing 11/100 and 80/100, clearly 80/100 is larger. So again, 9.8 is greater.

Wait, but just to make sure I haven't missed anything. Let me convert both numbers to the same denominator. If I have 9.11 = 9 + 0.11, and 0.11 is 11/100. 9.8 is 9 + 0.8, which is 9 + 8/10 or 9 + 40/50. But 11/100 is equivalent to 5.5/50, so comparing 5.5/50 and 40/50, 40/50 is much bigger. So indeed, 9.8 is larger.

Alternatively, maybe think in terms of money. If I have $9.11 and someone else has $9.80, who has more money? Well, $9.80 is more because after the decimal, 80 cents is more than 11 cents. So in real-world terms, that also makes sense.

Is there another way to verify? Maybe multiply both numbers by 100 to eliminate decimals. 9.11 * 100 = 911, and 9.8 * 100 = 980. Comparing 911 and 980, clearly 980 is larger. Therefore, 9.8 is greater than 9.11.

Hmm, all these methods point to the same conclusion. So even though 9.11 has two digits after the decimal and 9.8 has only one, the tenths place is more significant, so the 8 in the tenths place of 9.8 outweighs the 1 in the tenths place of 9.11, even with the extra hundredth. So 9.8 is definitely greater.

I think that's thorough enough. All the different methods confirm that 9.8 is greater than 9.11. No matter how I approach it—comparing place values, subtracting, converting to fractions, real-life analogies, or scaling up—the result is consistent. So I'm confident that 9.8 is the greater number.

**Final Answer**
\\boxed{9.8}<|END_THINKING|>To determine which number is greater between 9.11 and 9.8, we start by comparing their whole number parts, which are both 9. Next, we compare the tenths place. 

- For 9.11, the tenths place is 1.
- For 9.8, the tenths place is 8.

Since 8 is greater than 1, 9.8 is greater than 9.11. To verify, we can convert both numbers to the same format:
- Writing 9.8 as 9.80, we compare 9.11 and 9.80. The tenths place (8 vs. 1) confirms 9.80 is greater.

Subtracting 9.11 from 9.8:
\\[
     9.80 - 9.11 = 0.69
     \\]
     The result is positive, confirming 9.8 is greater.
     
     Converting to fractions:
     - 9.11 is \\(9 + \\frac{11}{100}\\)
     - 9.8 is \\(9 + \\frac{80}{100}\\)
     Comparing \\(\\frac{80}{100}\\) and \\(\\frac{11}{100}\\), we see \\(\\frac{80}{100}\\) is greater.
     
     Multiplying by 100:
     - 9.11 becomes 911
     - 9.8 becomes 980
     Comparing 911 and 980, 980 is greater.
     
     Thus, the final answer is \\(\\boxed{9.8}\\).
```

## Convert Command-A-Reasoning to FP8
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
                                        ignore = ['re:.*lm_head'],
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

## Command-A-Reasoning with FP8

```bash
# Start server with FP8 model on 2 H100_80GB GPUs for Command-A-Reasoning.
vllm serve path-to-CohereLabs/c4ai-command-a-reasoning-08-2025-fp8 \
     --tensor-parallel-size 2 \
     --enable-chunked-prefill
     --max-num-batched-tokens 4096
     --max-model-len 256000
     --gpu-memory-utilization 0.95
     --quantization=compressed-tensors
```
