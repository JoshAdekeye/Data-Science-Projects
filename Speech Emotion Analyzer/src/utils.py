import librosa
import numpy as np
import soundfile as sf

def load_audio(file_path, duration=3):
    """Load and preprocess audio file."""
    try:
        # Load audio file with specified duration
        data, sample_rate = librosa.load(file_path, duration=duration)
        
        # Resample if necessary
        if len(data) > sample_rate * duration:
            data = data[:sample_rate * duration]
        elif len(data) < sample_rate * duration:
            data = np.pad(data, (0, sample_rate * duration - len(data)))
            
        return data, sample_rate
    except Exception as e:
        print(f"Error loading audio file: {str(e)}")
        return None, None

def extract_features(data, sample_rate):
    """Extract audio features using librosa."""
    try:
        # MFCC (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        
        # Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=data, sr=sample_rate)
        mel_scaled = np.mean(mel.T, axis=0)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=data, sr=sample_rate)
        chroma_scaled = np.mean(chroma.T, axis=0)
        
        # Combine all features
        features = np.concatenate([mfccs_scaled, mel_scaled, chroma_scaled])
        return features
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

def record_audio(duration=3, sample_rate=22050):
    """Record audio from microphone."""
    import sounddevice as sd
    
    try:
        print("Recording...")
        audio_data = sd.rec(int(duration * sample_rate),
                          samplerate=sample_rate,
                          channels=1)
        sd.wait()
        print("Recording complete.")
        return audio_data.flatten(), sample_rate
    except Exception as e:
        print(f"Error recording audio: {str(e)}")
        return None, None

def save_audio(file_path, data, sample_rate):
    """Save audio data to file."""
    try:
        sf.write(file_path, data, sample_rate)
        return True
    except Exception as e:
        print(f"Error saving audio: {str(e)}")
        return False

def get_emotion_label(prediction):
    """Convert model prediction to emotion label."""
    emotions = {
        0: 'female_angry',
        1: 'female_calm',
        2: 'female_fearful',
        3: 'female_happy',
        4: 'female_sad',
        5: 'male_angry',
        6: 'male_calm',
        7: 'male_fearful',
        8: 'male_happy',
        9: 'male_sad'
    }
    return emotions.get(prediction, 'unknown') 