# backend.py (Final Deployment Ready Version)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
import requests
import json
from textblob import TextBlob
import collections
import time
import os
from dotenv import load_dotenv

# --- 1. INITIALIZE APP & LOAD ENVIRONMENT ---
# This correctly loads your API key from your 'key.env' file.
load_dotenv(dotenv_path='key.env')
app = Flask(__name__)
CORS(app)

# --- 2. LOAD THE TRAINED MODEL (The Correct, Simple Way) ---
print("Loading AI model (VGG16 on RAF-DB)...")
try:
    IMAGE_SIZE = 96
    emotion_model = load_model("VGG16_RAFDB_Pro_Model.h5")
    print("Model loaded successfully! Server is ready.")
except Exception as e:
    print(f"FATAL ERROR: Could not load model. Make sure 'VGG16_RAFDB_Pro_Model.h5' is in the folder. Error: {e}")
    exit()

# The correct "decoder ring" for the RAF-DB dataset's numerical labels.
emotion_labels = ['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']


# --- 3. PRO-GRADE STATE & TIMERS ---
prediction_history = collections.deque(maxlen=8)
conversation_history = []
last_interaction_time = time.time()
is_conversation_active = False

STABLE_DETECTIONS_REQUIRED = 5
# --- THIS IS THE FIX ---
# Lowered confidence threshold for a more responsive prototype
CONFIDENCE_THRESHOLD = 0.20
INITIAL_EMOTION_COOLDOWN = 4
IDLE_REENGAGE_TIMEOUT = 25
CONVERSATION_RESET_TIMEOUT = 60

# --- 4. HELPER FUNCTIONS ---
def get_text_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.2: return "Positive"
    elif analysis.sentiment.polarity < -0.2: return "Negative"
    else: return "Neutral"

def get_gemini_response(prompt):
    apiKey = os.getenv("GEMINI_API_KEY")
    if not apiKey:
        print("Error: GEMINI_API_KEY not found in .env file.")
        return "I'm sorry, my connection is not configured correctly."
    
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={apiKey}"
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(apiUrl, headers=headers, data=json.dumps(data), timeout=20)
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
    global last_interaction_time, conversation_history, is_conversation_active

    data = request.get_json()
    image_data = base64.b64decode(data['image'].split(',')[1])
    image = Image.open(io.BytesIO(image_data))
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    roi_color_resized = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    roi = preprocess_input(np.expand_dims(roi_color_resized.copy(), axis=0))
    
    prediction = emotion_model.predict(roi, verbose=0)[0]
    pred_prob = np.max(prediction)
    pred_label = emotion_labels[np.argmax(prediction)]
    
    if pred_prob >= CONFIDENCE_THRESHOLD:
        prediction_history.append(pred_label)

    ai_response = None
    prompt = None
    current_time = time.time()

    if not is_conversation_active and (current_time - last_interaction_time > INITIAL_EMOTION_COOLDOWN):
        if len(prediction_history) >= STABLE_DETECTIONS_REQUIRED:
            most_common_emotion = max(set(prediction_history), key=prediction_history.count)
            if prediction_history.count(most_common_emotion) >= STABLE_DETECTIONS_REQUIRED and most_common_emotion != 'neutral':
                emotion_prompts = {
                    'sad': "The user has been detected as sad. As an empathetic AI for youth mental wellness, start a gentle conversation.",
                    'angry': "The user has appeared angry. As an empathetic AI, gently ask what's on their mind.",
                    'happy': "The user has been smiling. As a friendly AI, share in their joy.",
                    'fear': "The user seems fearful. As a calming AI, gently reassure them.",
                    'surprise': "The user looks surprised. As a curious AI, ask them what happened.",
                    'disgust': "The user appears disgusted. As a supportive AI, ask if something is bothering them."
                }
                prompt = emotion_prompts.get(most_common_emotion)
                if prompt:
                    print(f"--- Proactive chat triggered by STABLE EMOTION: {most_common_emotion} ---")
                    is_conversation_active = True

    elif is_conversation_active and (current_time - last_interaction_time > IDLE_REENGAGE_TIMEOUT):
        last_emotion = prediction_history[-1] if prediction_history else 'neutral'
        prompt = (f"The user hasn't replied for over 25 seconds. Their current facial emotion is '{last_emotion}'. "
                  f"Gently re-engage them based on the conversation history: {json.dumps(conversation_history)}")
        print(f"--- Proactive chat triggered by IDLE ---")
    
    if is_conversation_active and (current_time - last_interaction_time > CONVERSATION_RESET_TIMEOUT):
        print("--- Conversation timed out due to inactivity. Resetting state. ---")
        is_conversation_active = False
        conversation_history = []
        prediction_history.clear()

    if prompt:
        ai_response = get_gemini_response(prompt)
        last_interaction_time = time.time()
        conversation_history.append({"ai": ai_response})
        prediction_history.clear()

    return jsonify({'emotion': pred_label, 'ai_response': ai_response})


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    global conversation_history, last_interaction_time
    data = request.get_json()
    user_message = data.get('message', '')
    facial_emotion = data.get('emotion', 'Neutral')
    conversation_history.append({"user": user_message})
    text_sentiment = get_text_sentiment(user_message)
    
    prompt = (f"You are Aura, an empathetic AI companion for youth mental wellness. "
              f"The user's facial emotion is '{facial_emotion}'. "
              f"The sentiment of their words is '{text_sentiment}'. "
              f"Recent history: {json.dumps(conversation_history)}. "
              f"The user just said: '{user_message}'. Respond thoughtfully.")
              
    ai_response = get_gemini_response(prompt)
    
    conversation_history.append({"ai": ai_response})
    if len(conversation_history) > 8:
        conversation_history = conversation_history[-8:]
    
    last_interaction_time = time.time()
    
    return jsonify({'reply': ai_response})

# --- 6. RUN THE APP ---
if __name__ == '__main__':
    # --- THIS IS THE FIX ---
    # This is the standard way to run a Flask app for deployment on services like Render
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

