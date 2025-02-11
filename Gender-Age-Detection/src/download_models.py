import os
import gdown
import requests

def download_models():
    """Download required model files."""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Model URLs
    model_urls = {
        # Face Detection Model
        'opencv_face_detector_uint8.pb': 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb',
        'opencv_face_detector.pbtxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/opencv_face_detector.pbtxt',
        
        # Age Detection Model
        'age_net.caffemodel': 'https://drive.google.com/uc?id=1kiusFljZc9QfcIYdU2s7xrtWHTraHwmW',
        'age_deploy.prototxt': 'https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_deploy.prototxt',
        
        # Gender Detection Model
        'gender_net.caffemodel': 'https://drive.google.com/uc?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ',
        'gender_deploy.prototxt': 'https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/gender_deploy.prototxt'
    }
    
    for filename, url in model_urls.items():
        output_path = os.path.join(models_dir, filename)
        if not os.path.exists(output_path):
            print(f"Downloading {filename}...")
            try:
                if 'drive.google.com' in url:
                    gdown.download(url, output_path, quiet=False)
                else:
                    response = requests.get(url)
                    response.raise_for_status()  # Raise an error for bad status codes
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                print(f"Successfully downloaded {filename}")
            except Exception as e:
                print(f"Error downloading {filename}: {str(e)}")
                print("Please download this file manually from the sources mentioned in README.md")
        else:
            print(f"{filename} already exists")

if __name__ == "__main__":
    download_models() 