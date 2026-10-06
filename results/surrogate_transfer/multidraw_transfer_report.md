# Multi-draw surrogate transfer

Models: GraySwanAI/Llama-3-8B-Instruct-RR, meta-llama/Llama-3.1-8B-Instruct
Draws: [0, 1]

Maximum truncated-thinking rate: 0.00%.
Maximum finish_reason=length rate: 27.36%.
Maximum degenerate-repetition rate: 26.53%.
Capability-valid under the preregistered 5% truncation ceiling: **True**.
Zero-length-truncation complete generation: **False**.
High truncation means the measured protocol is valid as a fixed-budget pipeline result but not as an estimate of unconstrained model capability.

## asr

Target rate: 66.61%

- GraySwanAI/Llama-3-8B-Instruct-RR: success 3.64%; repeat disagreement 6.35% (180,360 cells)
- meta-llama/Llama-3.1-8B-Instruct: success 74.97%; repeat disagreement 19.90% (180,360 cells)

| score | arm Spearman | target-cell AUC | top-10 target rate |
|---|---:|---:|---:|
| GraySwanAI/Llama-3-8B-Instruct-RR | -0.360 | 0.400 | 72.00% |
| meta-llama/Llama-3.1-8B-Instruct | 0.370 | 0.737 | 77.00% |
| ensemble | 0.299 | 0.719 | 79.00% |

## retained

Target rate: 9.69%

- GraySwanAI/Llama-3-8B-Instruct-RR: success 0.28%; repeat disagreement 0.53% (180,360 cells)
- meta-llama/Llama-3.1-8B-Instruct: success 4.37%; repeat disagreement 5.86% (180,360 cells)

| score | arm Spearman | target-cell AUC | top-10 target rate |
|---|---:|---:|---:|
| GraySwanAI/Llama-3-8B-Instruct-RR | 0.440 | 0.661 | 28.00% |
| meta-llama/Llama-3.1-8B-Instruct | 0.641 | 0.825 | 41.00% |
| ensemble | 0.638 | 0.823 | 35.00% |

## strict

Target rate: 9.53%

- GraySwanAI/Llama-3-8B-Instruct-RR: success 0.02%; repeat disagreement 0.03% (180,360 cells)
- meta-llama/Llama-3.1-8B-Instruct: success 3.76%; repeat disagreement 5.07% (180,360 cells)

| score | arm Spearman | target-cell AUC | top-10 target rate |
|---|---:|---:|---:|
| GraySwanAI/Llama-3-8B-Instruct-RR | 0.322 | 0.585 | 28.00% |
| meta-llama/Llama-3.1-8B-Instruct | 0.671 | 0.837 | 53.00% |
| ensemble | 0.671 | 0.838 | 53.00% |
