import os
import sys
from dotenv import load_dotenv
from livekit.agents import JobContext, JobProcess, WorkerOptions, cli
from livekit import agents
from livekit.plugins import openai, deepgram, cartesia, silero

load_dotenv(".env")

#   - vLLM   → usually http://<host>:8000/v1  (or your ngrok/Modal/Space URL + /v1)
#   - llama.cpp server → usually http://<host>:8001/v1

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL_NAME", "A7med-Ame3/Qwen2.5-7B-LiveKit-16bit")

SYSTEM_PROMPT = (
    "أنت شيخ مصري حكيم وطيب، تتحدث بالعامية المصرية البسيطة والمحببة للقلب. "
    "ترد على الأسئلة الدينية بشكل مباشر وميسر ومبسط، وتستخدم عبارات طيبة ودعائية. "
    "إجاباتك مختصرة وواضحة، من غير رموز أو إيموجي أو تنسيق معقد."
)

# 1) STT
# 2) Custom LLM
# 3) TTS


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    llm = openai.LLM(
        base_url=LLM_BASE_URL,
        api_key="EMPTY",
        model=LLM_MODEL_NAME,
        temperature=0.6,
    )

    agent = agents.Agent(instructions=SYSTEM_PROMPT)

    session = agents.AgentSession(
        stt=deepgram.STT(
            model="nova-3",
            language="ar",
        ),
        llm=llm,
        tts=cartesia.TTS(
            model="sonic-3",
            language="ar",
            voice="40f9b5d1-bc79-43a6-b5cc-1c692b3b40d2",
        ),
        vad=silero.VAD.load(),
    )

    await session.start(room=ctx.room, agent=agent)

    await session.say(
        "أهلاً بك! أنا هنا عشان أساعدك في أي سؤال ديني، اتفضل اسأل.",
        allow_interruptions=True,
    )


if __name__ == "__main__":
    sys.argv = ["agent.py", "dev"]
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint,
        agent_name="egyptian-dialect-agent")
    )
