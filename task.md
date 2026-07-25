# 🎯 AloEgy.ai Technical Assessment: Egyptian Dialect LLM Fine-Tuning

> **Sender:** Rokia Osama (Founder | AloEgy.ai)
> **Target Role / Focus:** LLM Fine-Tuning, Low-Latency Streaming, Multi-Agent Workflows
> **Timeline:** 5 to 7 Days

---

## 📌 Context & Objective

At **AloEgy.ai**, our core engine relies heavily on low-latency, specialized Large Language Models that natively understand and respond in **Egyptian Arabic dialect (العامية المصرية)** for real-time conversational workflows.

Your goal is to:

1. **Fine-tune** an open-source LLM specifically tailored for Egyptian Arabic dialog and intent extraction.
2. **Optimize & Deploy** it for real-time streaming inference.
3. **Integrate** it with the LiveKit Agents Framework.
4. **Benchmark** and thoroughly document/justify your technical methodology.

---

## 🛠️ Assessment Task Breakdown

### 1. Base Model & Dataset Selection

* **Base Model Selection:**
  Choose an appropriate open-source base LLM suitable for low-latency Arabic/multilingual inference (e.g., `Qwen 2.5`, `Llama 3 / 3.1`, `Gemma 2`, etc.).
* **Dataset Curation / Preparation (Domain of Your Choice):**

  * Select an existing open-source Egyptian Arabic dataset or curate/synthesize your own.
  * Pick any practical real-world scenario where fine-tuning for fluent Egyptian Arabic adds significant value (e.g., *e-commerce customer support, banking assistant, telecom helpdesk, or general Egyptian conversational AI*).
  * Ensure the dataset properly captures natural Egyptian phrasing, slang, intent extraction, and contextual nuances.

---

### 2. Model Fine-Tuning & Optimization

* **Fine-Tuning:**
  Fine-tune the model using Parameter-Efficient Fine-Tuning techniques (**PEFT / LoRA / QLoRA**). Acceleration libraries like `Unsloth`, `TRL`, or `DeepSpeed` are highly encouraged.
* **Inference Optimization:**
  Prepare the fine-tuned checkpoint for low-latency production deployment (e.g., Quantization via `GGUF`, `vLLM`, or `Ollama` for fast token Model Fine-Tuning & Optimization

### 3. Evaluation & Benchmarking

- Conduct a systematic evaluation comparing the Base Model vs. Fine-Tuned Model.

- Measure and report metrics such as:

- Qualitative Accuracy: Understanding of Egyptian slang, intent extraction, and dialect consistency.

- Inference Benchmarks: Latency (Time-To-First-Token - TTFT, Tokens Per Second - TPS), memory footprint (VRAM usage), and quantization loss.

### 4. Real-Time Streaming & LiveKit Integration

* Study the official [LiveKit Agents Framework Documentation](https://docs.livekit.io/agents/).
* Integrate your fine-tuned model checkpoint (or local API server like vLLM/Ollama) into a basic Python worker using `livekit-agents` to demonstrate real-time text/audio interaction loops.

---

## 📦 Deliverables & Requirements

### 1. GitHub Repository

Must contain clean, runnable code for:

* Fine-tuning scripts / notebooks (Unsloth, SFTTrainer, etc.).
* Dataset samples and preparation scripts.
* Evaluation and benchmarking scripts.
* The `livekit-agents` Python worker implementation.

### 2. Design Justification Document (`ARCHITECTURE.md` or PDF)

You must explicitly justify every technical decision:

* **Base Model:** Why did you select your specific base model over alternatives for Egyptian Arabic?
* **Dataset & Domain:** Why did you choose your specific scenario, and what was your dataset cleaning, structuring, and synthetic generation methodology?
* **Optimization Setup:** Why did you pick your optimization techniques (e.g., LoRA rank/alpha, Unsloth vs. Hugging Face Trainer, 4-bit/8-bit quantization)?
* **Benchmarking Proof:** How do your benchmarking results prove that fine-tuning improved both dialect accuracy and inference latency?

---

## ⏱️ Timeline & Submission

* **Deadline:** 5 to 7 days.
* **Support:** Direct email replies are open for any technical or LiveKit documentation clarifications.
