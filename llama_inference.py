from pyngrok import ngrok
import sys
import time
import subprocess
import os
# توجيه الكاش لمجلد محلي داخل المشروع قبل استدعاء باقي المكتبات
os.environ["HF_HOME"] = "./.hf_cache"


PORT = 8000
REPO_ID = "A7med-Ame3/Qwen2.5-7B-GGUF"
FILENAME = "Qwen2.5-7B-LiveKit-16bit.Q4_K_M.gguf"


def start_llama_server():
    print("🚀 جاري تشغيل llama-cpp server...")

    cmd = [
        sys.executable, "-m", "llama_cpp.server",
        "--hf_model_repo_id", REPO_ID,
        "--model", FILENAME,
        "--n_gpu_layers", "-1",
        "--host", "0.0.0.0",
        "--port", str(PORT)
    ]

    server_process = subprocess.Popen(cmd)

    print("⏳ انتظار تحميل الموديل والسيرفر...")
    time.sleep(10)

    try:
        public_url = ngrok.connect(PORT).public_url
        base_url = f"{public_url}/v1"

        print("\n" + "="*50)
        print("✅ السيرفر شغال بنجاح!")
        print(f"🔗 انسخ هذا الرابط وضعه في ملف .env كـ LLM_BASE_URL:")
        print(f"\n{base_url}\n")
        print("="*50 + "\n")

    except Exception as e:
        print(f"❌ حدث خطأ أثناء فتح Ngrok: {e}")
        server_process.terminate()
        return

    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 جاري إيقاف السيرفر و Ngrok...")
        ngrok.kill()
        server_process.terminate()


if __name__ == "__main__":
    start_llama_server()
