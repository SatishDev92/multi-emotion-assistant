🎭 Real-Time Multimodal Emotion-Aware Assistant

This project is a real-time multimodal smart assistant that detects your emotions from text, voice, and facial expressions, and responds appropriately with emotional intelligence.

🌟 Features

🎤 Voice Emotion Detection using MFCC + CNN

📝 Text Emotion Classification with HuggingFace Transformers

😊 Facial Emotion Detection using OpenCV and a trained CNN model

🤖 Emotion-Adaptive Responses powered by Microsoft's GODEL model

🔊 Speech Recognition & Text-to-Speech for natural voice interaction

📁 Folder Structure

emotion-aware-assistant/
│
├── face_emotion_detection/
│   └── predict_face_emotion.py
│   └── ff.h5                      # Facial emotion model
│
├── voice_emotion_detection/
│   └── vo.py
│   └── model_3.h5                # Voice emotion model
│
├── text_emotion_detection/
│   └── text_chatbot.py           # Text emotion classification + GODEL response
│
├── fusion.py                     # The main fusion logic for all modalities
├── model.ipynb                   # Training or exploration notebook (optional)
├── requirements.txt              # All required dependencies
├── README.md                     # You are here!

🚀 How to Run

1. Clone the Repository

git clone https://github.com/your-username/emotion-aware-assistant.git
cd emotion-aware-assistant

2. Install Dependencies

pip install -r requirements.txt

3. Run the Assistant

python fusion.py

🧠 Models Used

Text Emotion: j-hartmann/emotion-english-distilroberta-base

Chatbot: microsoft/GODEL-v1_1-base-seq2seq

Voice Emotion: Custom CNN trained on TESS dataset

Face Emotion: Custom CNN trained on FER2013-like images



🙌 Acknowledgements

HuggingFace for amazing transformer models

Microsoft GODEL for emotional responses

TensorFlow / Keras for deep learning models

OpenCV for real-time face detection

TESS Dataset for voice training


