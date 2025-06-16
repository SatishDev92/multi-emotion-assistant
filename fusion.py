import threading
from concurrent.futures import ThreadPoolExecutor
import cv2
from face_emotion_detection.predict_face_emotion import predict_face_emotion
from voice_emotion_detection.vo import predict_voice_emotion
from text_emotion_detection.text import (
    get_text_from_speech, 
    classify_emotion, 
    generate_response, 
    speak_response
)

def combine_emotions(face, voice, text):
    """Weighted emotion combination with priority to voice and text"""
    emotion_scores = {
        'happy': 0,
        'sad': 0,
        'angry': 0,
        'neutral': 0,
        'surprised': 0,
        'fearful': 0,
        'disgust': 0
    }
    
    # Weights (adjust based on which modality you trust more)
    emotion_scores[face] += 0.3
    emotion_scores[voice] += 0.4
    emotion_scores[text] += 0.3
    
    return max(emotion_scores.items(), key=lambda x: x[1])[0]

def conversation_cycle():
    print("\n" + "="*50)
    print("Starting new conversation cycle...")
    print("="*50 + "\n")
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Start face detection in parallel (webcam will open)
        face_future = executor.submit(predict_face_emotion)
        
        # Start voice recording and emotion detection
        print("\nSpeak now...")
        voice_future = executor.submit(predict_voice_emotion)
        text_future = executor.submit(get_text_from_speech)
        
        # Get results
        face_emotion = face_future.result()
        voice_emotion = voice_future.result()
        user_text = text_future.result()
    
    print(f"\nDetection Results:")
    print(f"Face: {face_emotion}")
    print(f"Voice: {voice_emotion}")
    
    text_emotion = classify_emotion(user_text)
    print(f"Text: {text_emotion} (from: '{user_text}')")
    
    final_emotion = combine_emotions(face_emotion, voice_emotion, text_emotion)
    print(f"\nCombined Emotion: {final_emotion}")
    
    response = generate_response(user_text, final_emotion)
    print(f"\nAI Response: {response}")
    speak_response(response)

def main():
    print("Starting Interactive Emotion-Aware Chatbot")
    print("Press Ctrl+C to exit at any time\n")
    
    try:
        while True:
            conversation_cycle()

            print("\nReady for next conversation...")
    except KeyboardInterrupt:
        print("\nClosing chatbot...")
    finally:
        # Ensure any open resources are closed
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()