import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from generate_response import generate_response  # Import the bot logic

app = FastAPI()

# Setup CORS (adjust origins as needed)
origins = [
    "http://localhost:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:3000",
    "http://172.25.176.1:3000",
    "https://chatbot-gamma-smoky.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    text: str

@app.post("/chat")
async def chat(user_input: UserInput):
    return StreamingResponse(generate_response(user_input.text), media_type="text/plain")

# Ensure the app binds to Render's required port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)