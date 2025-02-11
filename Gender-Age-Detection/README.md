# Gender and Age Detection 🎯

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview
This project implements a real-time gender and age detection system using deep learning models. It can detect faces in images or video streams and predict the gender and approximate age range of detected individuals. The system uses OpenCV's DNN module and pre-trained Caffe models for accurate predictions.

## ✨ Features
- 👤 Face detection using OpenCV DNN
- 👫 Gender classification (Male/Female)
- 📅 Age range prediction in 8 categories
- 🖼️ Support for both image and video input
- ⚡ Real-time processing capabilities
- 🌐 Web interface for easy interaction
- 📊 Visual results with bounding boxes and labels

## 🏗️ Project Structure
```
├── src/               # Source code files
│   ├── model.py      # Core model functionality
│   ├── utils.py      # Utility functions
│   └── app.py        # Streamlit web application
├── models/           # Pre-trained model files
├── data/            # Test images and sample data
├── requirements.txt  # Project dependencies
├── LICENSE          # MIT License
└── README.md        # Project documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip package manager
- Git (for cloning the repository)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gender-age-detection.git
cd gender-age-detection
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the required model files and place them in the `models/` directory:
- `opencv_face_detector_uint8.pb`
- `opencv_face_detector.pbtxt`
- `age_net.caffemodel`
- `age_deploy.prototxt`
- `gender_net.caffemodel`
- `gender_deploy.prototxt`

## 💻 Usage

### Command Line Interface
Process a single image:
```bash
python src/model.py --image data/your_image.jpg
```

### Web Interface
Launch the Streamlit app:
```bash
streamlit run src/app.py
```
Then open your browser and navigate to `http://localhost:8501`

## 🧠 Model Details

### Architecture
The system uses three deep learning models:
1. **Face Detection**: OpenCV face detector (DNN)
   - Architecture: Single Shot Detector (SSD)
   - Backend: TensorFlow
   
2. **Gender Classification**: Caffe model
   - Binary classification (Male/Female)
   - Trained on Adience dataset
   
3. **Age Detection**: Caffe model
   - 8 age range categories
   - Trained on Adience dataset

### Age Categories
The model predicts age in the following ranges:
- 👶 0-2 years
- 🧒 4-6 years
- 🧑 8-12 years
- 👱 15-20 years
- 👨 25-32 years
- 👨‍🦰 38-43 years
- 👨‍🦳 48-53 years
- 🧓 60-100 years

## 📊 Performance

The model performs best with:
- Well-lit faces
- Front-facing orientation
- Clear, unobstructed view
- Multiple faces in a single image
- Various age groups and genders

## 🛠️ Development

Want to contribute? Great! Here are some ways you can help:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 To-Do
- [ ] Add video stream support
- [ ] Implement batch processing
- [ ] Add confidence scores to predictions
- [ ] Create API endpoint
- [ ] Add more visualization options
- [ ] Improve age range accuracy

## 🔑 Requirements
- Python 3.7+
- OpenCV 4.5+
- NumPy
- Streamlit (for web interface)
- PIL (Python Imaging Library)

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- OpenCV for the face detection model
- Caffe model for age and gender detection
- Streamlit for the web interface framework
- The deep learning community for pre-trained models

## Model Files
The model files are not included in this repository due to size constraints. You can:

1. Download them automatically:
```bash
pip install gdown requests
python src/download_models.py
```

2. Or download manually and place in the `models/` directory:
- Face Detection: [OpenCV Face Detector](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)
- Age & Gender Models: [Adience Dataset Models](https://talhassner.github.io/home/projects/Adience/Adience-data.html)

Required files:
- opencv_face_detector_uint8.pb
- opencv_face_detector.pbtxt
- age_net.caffemodel
- age_deploy.prototxt
- gender_net.caffemodel
- gender_deploy.prototxt

---
Made with ❤️ using OpenCV and Streamlit 