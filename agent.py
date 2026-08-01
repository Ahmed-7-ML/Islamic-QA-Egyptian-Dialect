import os
import sys
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit import agents
from livekit.plugins import openai, deepgram, cartesia, silero

load_dotenv(".env")

# - vLLM ->  usually http://<host>:8000/v1  OR Ngrok Public URL + /v1)
# Run vLLM Server on Kaggle Notebook
# Make it Public via Ngork
# Take Public URL in Livekit Agent

# - llama.cpp -> 
# The Pipeline :-
# 1) STT        --> Convert my speech into text
# 2) Custom LLM --> Make Actions on a received text (Generate Answer)
# 3) TTS        --> Convert the Answer text into Speech again

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://unsignalised-englacial-vinnie.ngrok-free.dev/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "A7med-Ame3/Qwen2.5-7B-LiveKit-16bit")
# LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen2.5-7B-LiveKit-16bit.Q4_K_M.gguf")

SYSTEM_PROMPT = (
    # --> Context 
    "أنت شيخ مصري حكيم وطيب، اتخرجت من الأزهر الشريف، بتتكلم بالعامية المصرية "
    "البسيطة اللي بيفهمها كل الناس، وأسلوبك دافئ ومحبب للقلب زي شيخ الحتة اللي "
    "الناس بتحبه وترتاح تسأله. "

    # --> Instruction
    "لما حد يسألك سؤال ديني، جاوب إجابة مباشرة ومبسطة من غير تعقيد، واستشهد "
    "بآية أو حديث لو مناسب من غير ما تطوّل. لو السؤال فيه تفاصيل فقهية دقيقة أو "
    "خلافية بين المذاهب، وضّح إن الموضوع فيه تفصيل ونصح السائل يرجع لعالم "
    "متخصص أو دار الإفتاء للفتوى الدقيقة، بدل ما تجزم برأي واحد. "

    # --> Input Data
    "السؤال جالك من مستخدم بيتكلم معاك بالصوت وبيتحول لنص، فممكن يكون فيه "
    "كلمة غلط في التحويل أو السؤال مقطوع؛ لو حصل كده، اسأل توضيح بسيط بدل ما "
    "تخمن أو تجاوب على حاجة مش مفهومة. "

    # --> Output Indicator 
    "ردودك لازم تكون بالعامية المصرية بس، من جملتين لتلات جمل كحد أقصى في "
    "المرة الواحدة، من غير رموز أو إيموجي أو تنسيق زي النقط والعناوين، لأن "
    "كلامك هيتحول لصوت ولازم يتقال بشكل طبيعي متصل."
)

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
        # Try ElevenLabs with Egyptian Voice 
        tts=cartesia.TTS(
            model="sonic-3",
            language="ar",
            voice="40f9b5d1-bc79-43a6-b5cc-1c692b3b40d2",
        ),
        vad=silero.VAD.load(),
    )

    await session.start(room=ctx.room, agent=agent)

    await session.say(
        "اهلا بيك اخى العزيز,اسمك ايه و اقدر اساعدك ازاى ؟ ",
        allow_interruptions=True,
    )

if __name__ == "__main__":
    sys.argv = ["agent.py", "dev"]
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint,
        agent_name="egyptian-dialect-agent")
    )
