import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import tempfile

from utils import load_audio, extract_features, record_audio, get_emotion_label
from model import load_model

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                         'saved_models', 'emotion_model.h5')
model = load_model(model_path)

# Create templates directory and HTML files if they don't exist
os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)

# Create index.html
index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Speech Emotion Analyzer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 10px 0;
        }
        .button:hover {
            background-color: #45a049;
        }
        #result {
            margin-top: 20px;
            padding: 10px;
            border-radius: 5px;
        }
        .success {
            background-color: #dff0d8;
            border: 1px solid #d6e9c6;
            color: #3c763d;
        }
        .error {
            background-color: #f2dede;
            border: 1px solid #ebccd1;
            color: #a94442;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Speech Emotion Analyzer</h1>
        <div>
            <h3>Upload Audio File</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" name="audio" accept="audio/*" required>
                <button type="submit" class="button">Analyze</button>
            </form>
        </div>
        <div>
            <h3>Record Audio</h3>
            <button id="recordButton" class="button">Start Recording</button>
            <button id="stopButton" class="button" style="display: none;">Stop Recording</button>
        </div>
        <div id="result"></div>
    </div>

    <script>
        // File upload handling
        document.getElementById('uploadForm').onsubmit = function(e) {
            e.preventDefault();
            const formData = new FormData(this);
            
            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayResult(data);
            })
            .catch(error => {
                console.error('Error:', error);
                displayResult({error: 'Error processing audio file'});
            });
        };

        // Audio recording handling
        let mediaRecorder;
        let audioChunks = [];

        document.getElementById('recordButton').onclick = async function() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    const formData = new FormData();
                    formData.append('audio', audioBlob);

                    const response = await fetch('/analyze', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    displayResult(data);
                };

                mediaRecorder.start();
                document.getElementById('recordButton').style.display = 'none';
                document.getElementById('stopButton').style.display = 'inline-block';
            } catch (error) {
                console.error('Error:', error);
                displayResult({error: 'Error accessing microphone'});
            }
        };

        document.getElementById('stopButton').onclick = function() {
            mediaRecorder.stop();
            document.getElementById('recordButton').style.display = 'inline-block';
            document.getElementById('stopButton').style.display = 'none';
        };

        function displayResult(data) {
            const resultDiv = document.getElementById('result');
            if (data.error) {
                resultDiv.className = 'error';
                resultDiv.textContent = data.error;
            } else {
                resultDiv.className = 'success';
                resultDiv.textContent = `Detected Emotion: ${data.emotion}`;
            }
        }
    </script>
</body>
</html>
"""

# Write index.html
with open(os.path.join(os.path.dirname(__file__), 'templates', 'index.html'), 'w') as f:
    f.write(index_html)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded audio file."""
    temp_file = None
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'})
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        # Create temporary file with a unique name
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()  # Close the file handle immediately
        
        try:
            # Save uploaded file
            audio_file.save(temp_path)
            
            # Load and process audio
            data, sample_rate = load_audio(temp_path)
            if data is None:
                return jsonify({'error': 'Error processing audio file'})
            
            # Extract features
            features = extract_features(data, sample_rate)
            if features is None:
                return jsonify({'error': 'Error extracting features'})
            
            # Make prediction
            prediction = model.predict(np.array([features]))
            emotion = get_emotion_label(np.argmax(prediction[0]))
            
            return jsonify({'emotion': emotion})
            
        finally:
            # Clean up: ensure temporary file is deleted
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file {temp_path}: {str(e)}")
            
    except Exception as e:
        # If any error occurs, ensure we try to clean up
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except:
                pass
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 