import speech_recognition as sr
import pyttsx3
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------- SETUP --------------------
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 145)
voices = tts_engine.getProperty('voices')
tts_engine.setProperty('voice', voices[0].id)  # Change index if you want a different voice

print("📦 Loading models...")
device = 0 if torch.cuda.is_available() else -1  # for transformers pipeline device arg: 0 = GPU, -1 = CPU
print(f"Device set to: {'cuda' if device == 0 else 'cpu'}")

# Emotion classifier with device setting
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=device
)

# GODEL conversational model
chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
chat_model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
if device == 0:
    chat_model = chat_model.cuda()

print("✅ Models loaded. Ready to chat!")

# -------------------- EMOTION MAPPING --------------------
def map_emotion(label):
    mapping = {
        'anger': 'angry',
        'disgust': 'disgust',
        'fear': 'fearful',
        'joy': 'happy',
        'neutral': 'neutral',
        'sadness': 'sad',
        'surprise': 'surprised'
    }
    return mapping.get(label.lower(), 'neutral')

# -------------------- SPEECH TO TEXT --------------------
def get_text_from_speech():
    with sr.Microphone() as source:
        print("🎙️ Say something!")
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio)
            print(f"🗣️ You said: {text}")
            return text.lower()
        except sr.WaitTimeoutError:
            print("⏱️ Timeout: No speech detected.")
        except sr.UnknownValueError:
            print("❌ Could not understand audio.")
        except sr.RequestError:
            print("❌ API unavailable or unresponsive.")
        except Exception as e:
            print("❌ Error:", e)
    return ""

# -------------------- EMOTION DETECTION --------------------
def classify_emotion(text):
    if not text.strip():
        return "neutral"

    results = emotion_classifier(text)
    # results is a list of dicts like [{'label': 'joy', 'score': 0.8}, {...}]

    top_result = max(results, key=lambda x: x['score'])
    label = top_result['label']
    score = round(top_result['score'], 2)

    emotion = map_emotion(label)
    print(f"❤️ Detected Emotion: {emotion} ({score})")
    return emotion

# -------------------- EMOTION STYLE PROMPT --------------------
def build_prompt(user_text, emotion):
    style = {
        'happy': "Respond in a cheerful and supportive way to:",
        'sad': "Respond kindly and try to cheer them up for:",
        'angry': "Respond calmly and understandingly to:",
        'fearful': "Respond with comfort and reassurance to:",
        'neutral': "Respond informatively and helpfully to:",
        'disgust': "Respond politely and professionally to:",
        'surprised': "Respond enthusiastically and with curiosity to:"
    }
    return f"{style.get(emotion, 'Respond normally to:')} '{user_text}'"

# -------------------- CHAT RESPONSE GENERATION --------------------
def generate_response(user_text, emotion):
    prompt = build_prompt(user_text, emotion)
    input_text = f"Instruction: {prompt} \nInput: \nKnowledge:"
    inputs = chat_tokenizer(input_text, return_tensors="pt")
    if device == 0:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    output_ids = chat_model.generate(
        **inputs,
        max_length=128,
        do_sample=True,
        top_p=0.92,
        top_k=50,
        temperature=0.7,
        num_return_sequences=1
    )

    response = chat_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"🤖 Bot: {response}")
    return response

# -------------------- TEXT TO SPEECH --------------------
def speak_response(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# -------------------- MAIN LOOP --------------------
def TextChatbot_main():
    speak_response("🤖 Starting Emotion-Aware Chatbot. Say 'exit' to quit.")
    while True:
        user_text = get_text_from_speech()
        if not user_text:
            continue
        if "exit" in user_text or "quit" in user_text:
            print("👋 Exiting chatbot.")
            speak_response("Goodbye. Talk to you soon.")
            break

        emotion = classify_emotion(user_text)
        reply = generate_response(user_text, emotion)
        speak_response(reply)

if __name__ == "__main__":
    TextChatbot_main()