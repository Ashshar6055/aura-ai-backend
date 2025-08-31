# app.py (Final Version with All Fixes and Logging)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Input
import requests
import json
from textblob import TextBlob
import collections
import time
import os
from dotenv import load_dotenv

dotenv_path = r"C:\Users\91914\OneDrive\Desktop\wbs\pass.env"
load_dotenv(dotenv_path=dotenv_path)

# --- 1. INITIALIZE THE FLASK APP ---
app = Flask(__name__)
CORS(app)

# --- 2. LOAD THE AI MODEL ONCE AT STARTUP ---
print("Loading AI model...")
input_tensor = Input(shape=(48, 48, 3))
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(7, activation='softmax')(x)
emotion_model = Model(inputs=base_model.input, outputs=predictions)
emotion_model.load_weights('VGG16_finetuned.h5')
print("Model loaded successfully! Server is ready.")

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# --- 3. CONVERSATION STATE ---
emotion_history = collections.deque(maxlen=3) 
conversation_history = []
last_interaction_time = time.time()
is_ai_turn = False 

INITIAL_EMOTION_COOLDOWN = 5
IDLE_TIMEOUT = 30

# --- 4. HELPER FUNCTIONS ---
def get_text_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.2: return "Positive"
    elif analysis.sentiment.polarity < -0.2: return "Negative"
    else: return "Neutral"

def get_gemini_response(prompt):
    apiKey = os.getenv("GEMINI_API_KEY")
    if not apiKey:
        print("Error: GEMINI_API_KEY could not be loaded. Check your .env file.")
        return "I'm sorry, my connection to my core systems is not configured correctly."
        
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={apiKey}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(apiUrl, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        return result['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        print(f"Error calling API: {e}")
        return "I'm having a little trouble connecting right now."

# --- 5. API ENDPOINTS ---

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'ready', 'message': 'Aura AI is online.'})

@app.route('/predict_and_trigger', methods=['POST'])
def predict_and_trigger():
    global last_interaction_time, conversation_history, is_ai_turn

    data = request.get_json()
    image_data = base64.b64decode(data['image'].split(',')[1])
    image = Image.open(io.BytesIO(image_data))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    roi_color = cv2.resize(frame, (48, 48), interpolation=cv2.INTER_AREA)
    roi = roi_color.astype("float") / 255.0
    roi = np.expand_dims(roi, axis=0)
    prediction = emotion_model.predict(roi, verbose=0)[0]
    emotion_label = emotion_labels[np.argmax(prediction)]
    emotion_history.append(emotion_label)

    ai_response = None
    current_time = time.time()
    
    prompt = None
    if not is_ai_turn and (current_time - last_interaction_time > INITIAL_EMOTION_COOLDOWN):
        if len(emotion_history) == 3 and len(set(emotion_history)) == 1:
            trigger_emotion = emotion_history[0]
            if trigger_emotion != 'Neutral':
                emotion_prompts = {
                    'Sad': "The user has been sad for the last 5 seconds. Start a gentle and kind conversation.",
                    'Angry': "The user has looked angry for the last 5 seconds. Gently ask what's wrong.",
                    'Happy': "The user has been smiling for 5 seconds. Share in their joy and ask what's making them so happy."
                }
                prompt = emotion_prompts.get(trigger_emotion)
                if prompt:
                    print(f"--- Proactive chat triggered by EMOTION: {trigger_emotion} ---")

    elif is_ai_turn and (current_time - last_interaction_time > IDLE_TIMEOUT):
        last_emotion = emotion_history[-1]
        prompt = (f"The user hasn't replied for 30 seconds. Their current facial emotion is '{last_emotion}'. "
                  f"Gently continue the conversation or ask a follow-up question. "
                  f"Recent history: {json.dumps(conversation_history)}")
        if prompt:
            print(f"--- Proactive chat triggered by IDLE ---")
    
    if prompt:
        ai_response = get_gemini_response(prompt)
        last_interaction_time = time.time()
        conversation_history.append({"ai": ai_response})
        emotion_history.clear()
        is_ai_turn = True

    return jsonify({'emotion': emotion_label, 'ai_response': ai_response})


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    global conversation_history, is_ai_turn, last_interaction_time
    
    # NEW: Add print statements for debugging
    print("\n--- Received request in /chat endpoint ---")
    data = request.get_json()
    print(f"Raw data received: {data}")

    user_message = data.get('message', '')
    print(f"Extracted user message: '{user_message}'")
    
    facial_emotion = data.get('emotion', 'Neutral')
    text_sentiment = get_text_sentiment(user_message)
    
    conversation_history.append({"user": user_message})
    if len(conversation_history) > 4:
        conversation_history.pop(0)

    prompt = (f"Continue this conversation. The user's current facial emotion is '{facial_emotion}'. "
              f"The sentiment of their words is '{text_sentiment}'. "
              f"Recent conversation history: {json.dumps(conversation_history)}. "
              f"The user just said: '{user_message}'. Respond thoughtfully.")
    
    print(f"Prompt sent to Gemini: {prompt}")
              
    ai_response = get_gemini_response(prompt)
    
    conversation_history.append({"ai": ai_response})
    if len(conversation_history) > 4:
        conversation_history.pop(0)

    last_interaction_time = time.time()
    is_ai_turn = True 

    return jsonify({'reply': ai_response})


# --- 6. RUN THE APP ---
if __name__ == '__main__':
    # THE CRUCIAL FIX: Disable the reloader to stabilize global variables
    app.run(debug=True, use_reloader=False, port=5000)