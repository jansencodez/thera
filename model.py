import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

class TherapyBot:
    def __init__(self, data_path="data/output.json"):
        self.data_path = data_path
        self.tokenizer = Tokenizer(oov_token="<OOV>", lower=True)
        self.label_encoder = LabelEncoder()
        self.vocab_size = None
        self.max_length = 40  
        self.model = None
        
    def load_data(self):
        try:
            with open(self.data_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            # Handle both [...] formats
            if isinstance(data, dict) and "conversations" in data:
                data = data["conversations"]
            elif not isinstance(data, list):
                raise ValueError("Invalid JSON structure: Expected a list of conversations")

            if len(data) == 0:
                raise ValueError("Dataset is empty!")

            inputs, outputs = [], []
            for conv in data:
                if isinstance(conv, dict) and 'input' in conv and 'output' in conv:
                    inputs.append(conv['input'].strip())
                    outputs.append(conv['output'].strip())

            if len(inputs) != len(outputs):
                raise ValueError("Mismatched input/output pairs in dataset")

            # Calculate dynamic sequence length
            seq_lengths = [len(text.split()) for text in inputs]
            self.max_length = int(np.percentile(seq_lengths, 95))

            # Tokenization with emergency tokens
            self.tokenizer.fit_on_texts(inputs + ["[CRISIS]", "[UNK]"])
            self.vocab_size = len(self.tokenizer.word_index) + 1

            # Sequence padding with dynamic length
            X_train = pad_sequences(
                self.tokenizer.texts_to_sequences(inputs),
                maxlen=self.max_length,
                padding="post",
                truncating="post"
            )

            # Label encoding with class weights
            self.label_encoder.fit(outputs)
            y_train = self.label_encoder.transform(outputs)

            return X_train, np.array(y_train)
        except Exception as e:
            raise RuntimeError(f"Data loading failed: {str(e)}")
    
    def build_model(self):
        self.model = Sequential([
            Embedding(self.vocab_size, 128, input_length=self.max_length),
            Dropout(0.3),
            LSTM(128, return_sequences=True),
            LSTM(64),
            Dense(128, activation="relu"),
            Dropout(0.2),
            Dense(len(self.label_encoder.classes_), activation="softmax")
        ])
        
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                           loss="sparse_categorical_crossentropy",
                           metrics=["accuracy"])
    
    def train_model(self, X_train, y_train):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=20, restore_best_weights=True)
        self.model.fit(X_train, y_train, epochs=500, batch_size=16, callbacks=[early_stop], verbose=1)
    
    def save_model(self):
        self.model.save("saved_model/model.keras")
        with open("saved_model/tokenizer.pkl", "wb") as f:
            pickle.dump(self.tokenizer, f)
        with open("saved_model/label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)
    
    def load_model(self):
        self.model = tf.keras.models.load_model("saved_model/model.keras")
        with open("saved_model/tokenizer.pkl", "rb") as f:
            self.tokenizer = pickle.load(f)
        with open("saved_model/label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)
    
    def generate_response(self, user_input):
        crisis_keywords = {"suicide", "kill myself", "end it all", "self-harm", "want to die"}
        crisis_response = ("I'm deeply concerned about your safety. "
                           "Please contact the National Suicide Prevention Lifeline at 988 "
                           "or reach out to a trusted professional immediately."
                           "For more support you can reach out to the following:\nRed Cross Kenya :0700395395 / 1199 \nBefrienders Kenya:0722178177 \nRed Cross gender based violence helpline: 0800720745 \nChiromo Hospital group : https//:chiromohospitalgroup.co.ke \nMental 360: https//:mental360.co.ke/"
                           )
        
        if any(keyword in user_input.lower() for keyword in crisis_keywords):
            return crisis_response
        
        sequence = pad_sequences(self.tokenizer.texts_to_sequences([user_input]),
                                 maxlen=self.max_length, padding="post", truncating="post")
        prediction = self.model.predict(sequence, verbose=0)
        return self.label_encoder.inverse_transform([np.argmax(prediction)])[0]
    
if __name__ == "__main__":
    bot = TherapyBot()
    X_train, y_train = bot.load_data()
    bot.build_model()
    bot.train_model(X_train, y_train)
    bot.save_model()