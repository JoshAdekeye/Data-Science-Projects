# Speech Emotion Analyzer

## Overview
A deep learning model that detects emotions from speech using audio files. The model can:
- Detect emotions from audio files or live recordings
- Distinguish between male and female voices
- Classify emotions into categories: neutral, happy, sad, angry, and fearful
- Provide real-time analysis through a web interface

## Features
- Audio emotion detection with >70% accuracy
- Gender detection with high accuracy
- Support for both file upload and live recording
- Web-based user interface
- Real-time processing and results

## Project Structure
```
Speech-Emotion-Analyzer/
├── src/
│   ├── model.py        # CNN model architecture
│   ├── train.py        # Training script
│   ├── app.py          # Flask web application
│   └── utils.py        # Audio processing utilities
├── data/
│   └── RAVDESS/       # Dataset directory
├── saved_models/       # Trained model checkpoints
├── images/            # Documentation images
├── requirements.txt   # Project dependencies
├── LICENSE           # License information
└── README.md         # Project documentation
```

## Dataset
The model is trained on the RAVDESS dataset:
- [RAVDESS](https://zenodo.org/record/1188976): ~1500 audio files from 24 actors (12 male, 12 female)
- 8 different emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- File naming format includes emotion labels

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Speech-Emotion-Analyzer.git
cd Speech-Emotion-Analyzer
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
- Download RAVDESS dataset from [here](https://zenodo.org/record/1188976)
- Extract files to `data/RAVDESS/Audio_Song_Actors_01-24/`

## Usage

1. Training the model:
```bash
python src/train.py
```
This will:
- Process the audio files from the dataset
- Extract audio features
- Train the CNN model
- Save the trained model to `saved_models/`

2. Running the web application:
```bash
python src/app.py
```
Then open your browser and go to `http://localhost:5000`

## Web Interface Features
- Upload audio files for analysis
- Record audio directly through the browser
- Real-time emotion detection
- Clear visualization of results

## Emotion Labels
The model outputs predictions in the following format:
```
0 - female_angry
1 - female_calm
2 - female_fearful
3 - female_happy
4 - female_sad
5 - male_angry
6 - male_calm
7 - male_fearful
8 - male_happy
9 - male_sad
```

## Model Architecture
- Convolutional Neural Network (CNN)
- Input: Audio features (MFCC, Mel Spectrogram, Chroma)
- Multiple convolutional layers with batch normalization
- Dense layers with dropout for regularization
- Output: 10 emotion classes (5 emotions × 2 genders)

## Performance
- Gender detection accuracy: ~100%
- Emotion detection accuracy: >70%
- Real-time processing capability
- Low latency predictions

## Requirements
- Python 3.8+
- TensorFlow 2.x
- librosa
- Flask
- Other dependencies in requirements.txt

## License
This project is licensed under the terms of the LICENSE file included in the repository.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
