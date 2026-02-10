from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

app = Flask(__name__)
app.secret_key = "harvest_secret_key"

# Files to store data
COMMENTS_FILE = 'comments.json'
MESSAGES_FILE = 'messages.json'
USERS_FILE = 'users.json'

# Model Loading
model = tf.keras.models.load_model('plant_disease_model.keras')

CLASS_NAMES = [
    "Tomato - Healthy", "Tomato - Early Blight",
    "Potato - Healthy", "Potato - Late Blight",
    "Corn - Healthy", "Corn - Northern Leaf Blight",
    "Strawberry - Healthy", "Strawberry - Leaf Scorch",
    "Brinjal - Healthy", "Brinjal - Phomopsis Blight"
]

# Helper functions
def get_data(file_path):
    if not os.path.exists(file_path): return []
    with open(file_path, 'r') as f:
        try:
            return json.load(f)
        except:
            return []

def save_data(file_path, new_entry):
    data = get_data(file_path)
    data.append(new_entry)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# --- AUTH ROUTES ---
@app.route('/login.html')
def login_page():
    return render_template('login.html')

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    users = get_data(USERS_FILE)
    if any(u['email'] == data['email'] for u in users):
        return jsonify({"status": "error", "message": "Email already exists"}), 400
    save_data(USERS_FILE, data)
    return jsonify({"status": "success"})

@app.route('/signin', methods=['POST'])
def signin():
    data = request.get_json()
    users = get_data(USERS_FILE)
    user = next((u for u in users if u['email'] == data['email'] and u['password'] == data['password']), None)
    if user:
        session['user'] = user['name']
        return jsonify({"status": "success"})
    return jsonify({"status": "error", "message": "Invalid credentials"}), 401

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

# --- PAGE ROUTES ---
@app.route('/')
def index():
    comments = get_data(COMMENTS_FILE)
    return render_template('index.html', comments=comments, current_user=session.get('user'))

@app.route('/analysis-guide.html')
def analysis_guide(): return render_template('analysis-guide.html')

@app.route('/about.html')
def about(): return render_template('about.html')

@app.route('/contact.html')
def contact(): return render_template('contact.html')

# --- API ROUTES ---
@app.route('/add_comment', methods=['POST'])
def add_comment():
    if 'user' not in session:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401
    data = request.get_json()
    if data.get('text'):
        save_data(COMMENTS_FILE, {"name": session['user'], "text": data['text']})
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.get_json()
    if data.get('message'):
        save_data(MESSAGES_FILE, data)
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 400

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    result_idx = np.argmax(predictions[0])
    return jsonify({
        'prediction': CLASS_NAMES[result_idx],
        'confidence': f"{float(np.max(predictions[0]) * 100):.2f}%"
    })

if __name__ == '__main__':
    app.run(debug=True)