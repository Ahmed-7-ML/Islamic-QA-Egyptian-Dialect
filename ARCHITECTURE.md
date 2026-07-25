### Base Model Selection

### Why Qwen2.5-7B-Instruct over alternatives for Egyptian Arabic?

**Candidates evaluated** (via the [Open Arabic LLM Leaderboard (OALL)](https://huggingface.co/spaces/OALL/Open-Arabic-LLM-Leaderboard-v1), filtered to the ~7B parameter range for compute feasibility):

| Model                                                                                                            | OALL Average                                | Source              | License        | Verdict                                                                                                      |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------- | ------------------- | -------------- | ------------------------------------------------------------------------------------------------------------ |
| **Qwen2.5-7B-Instruct**                                                                                    | **54.3**                              | Official (Qwen)     | Apache 2.0     | ✅ Selected                                                                                                  |
| Qwen3-8B                                                                                                         | Not benchmarked on OALL at time of decision | Official (Qwen)     | Apache 2.0     | Rejected — no verifiable Arabic benchmark data available                                                    |
| Various "abliterated"/"uncensored" Qwen2.5-7B fine-tunes (e.g.,`huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2`) | 52-55                                       | Community           | Varies         | Rejected — safety alignment deliberately removed, unsuitable for a production-style assistant               |
| `MaziyarPanahi/calme-2.1-qwen2.5-72b`                                                                          | 70.62                                       | Community fine-tune | Tongyi Qianwen | Rejected — 73B parameters infeasible for available GPU memory and project timeline                          |
| `Qwen/Qwen3-VL-8B-Instruct`                                                                                    | N/A                                         | Official (Qwen)     | Apache 2.0     | Rejected — vision-language model; text-only understanding was required, adding unused multimodal complexity |

**Decision rationale:**

1. **Verified Arabic performance, not assumed.** Qwen2.5-7B-Instruct is the highest-scoring officially-released, unmodified model within a computationally feasible size range on OALL — a decision grounded in measured benchmark data rather than reputation or recency.
2. **Correct model type.** Confirmed as a pure text-generation ("Causal LM") model, avoiding the mistake of selecting a vision-language, coder, or omni-modal variant that would add unused complexity and resource overhead for a text-only conversational task.
3. **Official, safety-aligned release.** Rejected multiple higher-scoring community fine-tunes because they were explicitly "abliterated" (safety guardrails removed), which is inappropriate for an assistant intended for real users, even in a technical assessment context.
4. **Resource feasibility.** At ~7B parameters, the model fits within free/consumer-tier GPU memory for QLoRA fine-tuning (~7.6 GiB for full weights, confirmed empirically during training), within the project's multi-day timeline.
5. **Framework maturity.** Full compatibility with Unsloth (fast QLoRA training), vLLM, and llama.cpp/GGUF — all three deployment paths used in this project — with no compatibility issues encountered.

---

## Dataset Curation / Preparation

### Domain Selected: Islamic Q&A (General Egyptian Conversational AI)

### Datasets Considered

| Dataset                                                    | Format                                                                        | Size                 | License                  | Verdict              |
| ---------------------------------------------------------- | ----------------------------------------------------------------------------- | -------------------- | ------------------------ | -------------------- |
| **`Omar-youssef/islamic-qa-egyptian-arabic` (HF)** | **Structured Q&A pairs**                                                | **7,465 rows** | **Apache 2.0**     | **✅Selected** |
| `Sant0s3/Egyptian_dialect-filtered-data` (HF)            | Raw ASR transcript (single-speaker monologue)                                 | 3.8K rows            | Unclear                  | ❌ Rejected          |
| `Kyrillos2001/Egyptian_Dialect` (HF)                     | Raw ASR transcript (single-speaker monologue)                                 | 2.4K rows            | Unclear                  | ❌ Rejected          |
| Kaggle "2.5M Egyptian Datasets Collection"                 | Raw unstructured corpus (tweets, comments, lyrics, articles from 12+ sources) | 2.5M+ rows           | Mixed/unclear per-source | ❌ Rejected          |

---

## Model Fine-Tuning & Optimization

### Training Configuration

- **Base model**: `unsloth/Qwen2.5-7B-Instruct`, loaded in 4-bit (QLoRA) with `max_seq_length=2048`.
- **LoRA setup**: rank `r=16`, `lora_alpha=16` (alpha = rank, per standard guidance), targeting all attention and MLP projection layers (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`), with `use_rslora=True` (Rank-Stabilized LoRA) for more stable training at this rank, and `lora_dropout=0`.
- **Framework**: Unsloth + TRL's `SFTTrainer`, chosen over a standard Hugging Face `Trainer` for its significantly faster training throughput and lower VRAM footprint on limited compute (free-tier GPUs), which was necessary given the project timeline.
- **Training format**: Each example is formatted using the tokenizer's native chat template (`apply_chat_template`) with `user`/`assistant` roles, rather than a custom delimiter format. This ensures the training distribution exactly matches the inference-time prompt format, avoiding train/inference mismatch.
- **Hyperparameters**: batch size 2 with gradient accumulation of 4 (effective batch size 8), 2 epochs, learning rate 2e-4 with linear decay, `adamw_8bit` optimizer (memory-efficient), fp16 precision (appropriate for the training GPU, which lacks native bf16 support).
- **Data split**: 90/10 train/eval split (`seed=3407`) performed before training, so evaluation during training reflects held-out performance rather than memorized examples.

### Intent Labeling Methodology

Each training example was tagged with a heuristic intent category (`aqeedah`, `fiqh`, `history`, `general`) derived from keyword matching against the dataset's `source_topics` field. This labeling serves two purposes:

1. **Dataset stratification** — enabling evaluation across distinct question types rather than treating the dataset as a monolithic block.
2. **Evaluation reference** — providing a basis for constructing a diverse, representative evaluation question set.

**Design clarification**: The fine-tuning objective itself (the `question → answer` format used in training) does not include intent labels as part of the model's output. The model is trained to answer questions fluently in Egyptian dialect; intent classification is used as a dataset-curation and evaluation tool, not as a task the fine-tuned model is explicitly trained to perform as structured output. This distinction is intentional and documented here to avoid ambiguity about model capabilities.

### Deployment Artifacts Produced

| Artifact            | Purpose                                                     | Location                                |
| ------------------- | ----------------------------------------------------------- | --------------------------------------- |
| LoRA adapters       | Lightweight, shareable fine-tune                            | `A7med-Ame3/qwen2.5_lora_model`       |
| Merged 16-bit model | Full-precision model for vLLM serving                       | `A7med-Ame3/Qwen2.5-7B-LiveKit-16bit` |
| GGUF (q4_k_m)       | Quantized model for low-latency local inference (llama.cpp) | `A7med-Ame3/Qwen2.5-7B-Sheikh-GGUF`   |

## Optimization Setup Justification

### Why LoRA rank=16, alpha=16?

- **Rank (r=16)**: A mid-range rank was chosen as a balance between adaptation capacity and training efficiency. Given the dataset size (~6,700 training examples after the 90/10 split) and the fine-tuning goal (dialect/style adaptation + domain Q&A, not teaching entirely new capabilities), a rank in the 8-32 range is standard practice; 16 was selected as a safe middle ground that avoids both underfitting (too low a rank to capture stylistic shift) and unnecessary parameter overhead (a much higher rank with diminishing returns for a dataset this size).
- **Alpha (lora_alpha=16, i.e., alpha = rank)**: This follows the widely-used heuristic of setting alpha equal to rank when combined with **Rank-Stabilized LoRA (`use_rslora=True`)**. RSLoRA changes the scaling factor to `alpha / sqrt(r)` instead of the vanilla `alpha / r`, which keeps gradient magnitudes stable at higher ranks without requiring the alpha = 2×rank convention sometimes used with vanilla LoRA. This combination was chosen specifically to allow stable training at rank 16 without manual scaling tuning.
- **Target modules**: All attention projections (`q_proj, k_proj, v_proj, o_proj`) and all MLP projections (`gate_proj, up_proj, down_proj`) were adapted, rather than attention-only, to give the adapter more capacity to shift both reasoning/attention patterns and token-level output style (needed for dialect adaptation, which is largely a surface-realization change reflected in the MLP/output layers).

### Why Unsloth over the standard Hugging Face Trainer?

- **Training speed and memory**: Unsloth's custom kernels reduce VRAM usage and increase training throughput substantially compared to a standard `transformers.Trainer` + PEFT setup, on the same hardware.
- **4-bit QLoRA integration**: Unsloth provides a streamlined `FastLanguageModel.from_pretrained(..., load_in_4bit=True)` path with built-in gradient checkpointing (`use_gradient_checkpointing='unsloth'`), reducing implementation complexity compared to manually configuring `bitsandbytes` + `peft` + gradient checkpointing separately with a vanilla Trainer.
- **Native GGUF/merge export**: Unsloth's `save_pretrained_gguf` / `push_to_hub_merged` methods directly produced both required deployment artifacts (merged 16-bit model and GGUF quantized model) without needing a separate conversion toolchain, streamlining the path from training to the two required serving formats (vLLM and llama.cpp).

### Why 4-bit quantization during training, and q4_k_m for the GGUF deployment artifact?

- **4-bit (QLoRA) during training**: Loading the base model in 4-bit precision (`load_in_4bit=True`) reduces the memory footprint of the frozen base weights during fine-tuning, allowing a 7B-parameter model to be trained on GPUs with limited VRAM (the constraint driving nearly every infrastructure decision in this project, as documented in the deployment/debugging history). LoRA adapters themselves are still trained in higher precision, so this does not meaningfully degrade adaptation quality — it only reduces the memory cost of holding the frozen base weights.
- **q4_k_m for GGUF export**: Among Unsloth's supported quantization methods (`q8_0`, `q4_k_m`, `q5_k_m`), `q4_k_m` was selected as the recommended balance point — it keeps higher precision (Q6_K) for the attention output and half of the feed-forward weights (the layers most sensitive to quality loss) while using more aggressive Q4_K quantization elsewhere, giving a substantially smaller file size and faster CPU/edge inference than `q8_0` with comparatively minor quality trade-off, which matters directly for the "low-latency production deployment" requirement of this task.

---

## Evaluation & Benchmarking

### Methodology

Both the unmodified base model (`unsloth/Qwen2.5-7B-Instruct`) and the fine-tuned model were benchmarked using an identical prompt, decoding configuration, and measurement harness (`LatencyStreamer`, timing `time.time()` calls around `model.generate`), ensuring the comparison isolates the effect of fine-tuning rather than differences in test setup.

**Metrics measured:**

- **TTFT (Time-To-First-Token)**: latency from request start to first generated token — critical for real-time voice streaming responsiveness.
- **Total Inference Time**: full generation duration.
- **Output Token Count**: length of the generated response.
- **TPS (Tokens Per Second)**: `output_token_count / total_inference_time`.
- **Qualitative accuracy**: manual review of dialect fluency, directness, and correctness of the answer.

### Preliminary Result: Base vs. Fine-Tuned

**Test prompt** (Arabic, fiqh/aqeedah-adjacent intent): *"ما الفرق بين النبي و الرسول؟"* ("What is the difference between a Nabi and a Rasul?")

| Metric               | Base Model (`Qwen2.5-7B-Instruct`) | Fine-Tuned Model    | Change                 |
| -------------------- | ------------------------------------ | ------------------- | ---------------------- |
| TTFT                 | 11.38 sec (11,383 ms)                | 3.36 sec (3,356 ms) | **~3.4x faster** |
| Total Inference Time | 11.38 sec                            | 3.36 sec            | **~3.4x faster** |
| Output Tokens        | 128                                  | 47                  | ~63% shorter           |
| TPS                  | 11.24 tok/sec                        | 14.01 tok/sec       | ~25% higher            |

**Qualitative comparison:**

- **Base model output** (128 tokens, truncated mid-generation at `max_new_tokens` limit): produces a longer, more formal, MSA-leaning (Modern Standard Arabic) explanation with a numbered-list structure, and the response was cut off before completing the second point — indicating the base model did not converge to a concise answer within the token budget.
- **Fine-tuned model output** (47 tokens, complete): produces a short, direct answer in natural Egyptian dialect (e.g., "بيدعو للتوحيد وبيبلغ رسالته"), fully answering the question without truncation.

**Interpretation:**

1. The fine-tuned model produces substantially shorter, more conversational responses in this instance, which directly explains both the lower total inference time and the higher effective TPS — the model has learned the dataset's answer style (concise, dialect-driven, direct), rather than the base model's longer, more encyclopedic default style.
2. The TTFT improvement is consistent with generating a shorter, more confidently-decoded response rather than a change in raw per-token decoding speed alone (note TPS only improved ~25%, while total time improved ~3.4x — the gap is explained by the ~63% reduction in generated tokens).
3. The fine-tuned response uses natural Egyptian phrasing ("بيدعو", "بيكون قد") consistent with the target dialect, while the base model's response leans toward Modern Standard Arabic phrasing and formal listing structure — supporting the intended dialect-adaptation goal of fine-tuning.

### Limitations of This Result (To Be Addressed)

This comparison is based on **a single test prompt** and should be treated as preliminary, not conclusive:

- Sample size of one cannot rule out prompt-specific variance (e.g., caching effects, this specific question happening to trigger a longer/shorter base-model response by chance).
- No coverage yet across all four intent categories (`aqeedah`, `fiqh`, `history`, `general`) used in dataset curation.
- No quantization-loss comparison yet between the full-precision merged model and the GGUF `q4_k_m` quantized version.

**Next step**: Run the same base-vs-fine-tuned benchmark harness across a fixed set of 10–15 held-out questions spanning all four intent categories, and report aggregate (mean/median) TTFT, TPS, and a qualitative accuracy rubric score, rather than a single anecdotal example.
