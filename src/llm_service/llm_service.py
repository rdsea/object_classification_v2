
import os

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from ollama import AsyncClient

load_dotenv()

app = FastAPI()

ollama_host = os.getenv("OLLAMA_HOST")
llm_model = os.getenv("LLM_MODEL")


@app.post("/inference")
async def inference(request: Request):
    image_bytes = await request.body()

    client = AsyncClient(host=ollama_host)
    response = await client.generate(
        model=llm_model,
        prompt="what is in the image?",
        images=[image_bytes],
        stream=False,
    )
    return {"response": response["response"]}
