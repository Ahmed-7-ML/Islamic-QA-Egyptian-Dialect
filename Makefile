vllm: 
	python3 -m vllm.entrypoints.openai.api_server \
    --model A7med-Ame3/Qwen2.5-7B-LiveKit-16bit \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.80 \
    --enforce-eager \
    --port 8000 \

ubuntu:
	wsl -d Ubuntu

lk-run:
	lk agent deploy

HOST = 0.0.0.0
PORT = 8000
HF_REPO_ID = A7med-Ame3/Qwen2.5-7B-GGUF
MODEL_FILE = Qwen2.5-7B-LiveKit-16bit.Q4_K_M.gguf

llama-cpp:
	python -m llama_cpp.server \
		--hf_model_repo_id $(HF_REPO_ID) \
		--model $(MODEL_FILE) \
		--n_gpu_layers -1 \
		--host $(HOST) \
		--port $(PORT)
