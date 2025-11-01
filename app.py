import os
from flask import Flask, request, jsonify, render_template
from google import genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Flask and Gemini Setup ---
app = Flask(__name__)
# Get API Key from environment
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("FATAL: GEMINI_API_KEY not found. Please create a .env file and get a key from Google AI Studio.")
    exit()

client = genai.Client(api_key=API_KEY)
model_name = "gemini-2.5-flash"


# --- SYSTEM INSTRUCTION: ENFORCES POINT-BY-POINT ANSWERS ---
system_instruction = (
    "You are a friendly, helpful, and concise chatbot. Your answers must be "
    "formatted as a list of bullet points using standard Markdown (*). "
    "For simple greetings (e.g., 'Hi', 'Hello', 'Thank you'), respond with a "
    "short, friendly greeting or acknowledgment. "
    "For all other questions, provide an explanation using **no more than 12 concise bullet points**."
)

# Initialize the chat session with the system instruction
chat = client.chats.create(
    model=model_name,
    config=genai.types.GenerateContentConfig(
        system_instruction=system_instruction
    )
)


# --- Routes ---

@app.route('/')
def index():
    """Serves the main chatbot HTML page."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Handles the chat message from the frontend, sends it to Gemini, and returns the response."""
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Send message to the chat session
        response = chat.send_message(user_message)
        
        # The response text is returned in Markdown format
        return jsonify({"response": response.text})

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return jsonify({"error": "An error occurred during API call."}), 500

if __name__ == '__main__':
    app.run(debug=True)
