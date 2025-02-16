from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from generate_response import generate_response  # Import the bot logic

app = FastAPI()

# Setup CORS (adjust origins as needed)
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    text: str

@app.post("/chat")
async def chat(user_input: UserInput):
    return StreamingResponse(generate_response(user_input.text), media_type="text/plain")
