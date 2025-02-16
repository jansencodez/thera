import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import asyncio

# Load the updated model
model = tf.keras.models.load_model("saved_model/model.keras")

# Load the updated tokenizer
with open("saved_model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Load the updated label encoder
with open("saved_model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Define crisis keywords and crisis response
crisis_keywords = {"suicide", "kill myself", "end it all", "self-harm", "want to die"}
crisis_response = (
    "I'm deeply concerned about your safety. "
    "Please contact the National Suicide Prevention Lifeline at 988 "
    "or reach out to a trusted professional immediately."
)

async def generate_response(user_input: str):
    """
    Asynchronously generate a response for the given user input.
    Yields the response piece-by-piece for streaming.
    """
    user_input_lower = user_input.lower()

    # Crisis detection
    if any(keyword in user_input_lower for keyword in crisis_keywords):
        yield crisis_response
        return

    # Convert input text to sequence for prediction
    sequence = pad_sequences(
        tokenizer.texts_to_sequences([user_input_lower]),
        maxlen=20,
        padding="post",
        truncating="post"
    )
    prediction = model.predict(sequence, verbose=0)
    response_text = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    # Yield the response word-by-word with a slight delay for smooth streaming
    for word in response_text.split():
        yield word + " "
        await asyncio.sleep(0.05)
