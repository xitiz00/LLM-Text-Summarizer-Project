from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import requests
import json
import time
import logging
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='.')
# Enable CORS for all routes with more explicit settings
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Default to OpenAI if no Anthropic key
DEFAULT_MODEL = "anthropic" if ANTHROPIC_API_KEY else "openai"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
            
        text = data['text']
        length = data.get('length', 'medium')
        style = data.get('style', 'default')
        model = data.get('model', DEFAULT_MODEL)
        
        # Validate inputs
        if len(text) < 100:
            return jsonify({"error": "Text too short to summarize"}), 400
            
        # Select appropriate model
        if model == "anthropic" and ANTHROPIC_API_KEY:
            summary = summarize_with_anthropic(text, length, style)
        else:
            summary = summarize_with_openai(text, length, style)
            
        return jsonify({"summary": summary})
        
    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        return jsonify({"error": "Failed to generate summary", "details": str(e)}), 500

def generate_prompt(text, length, style):
    """Generate the prompt based on user settings"""
    
    # Length instructions
    length_instructions = {
        "short": "Create a very concise summary in 2-3 sentences.",
        "medium": "Create a medium-length summary in 4-6 sentences capturing the main points.",
        "long": "Create a comprehensive summary in 7-10 sentences."
    }
    
    # Style instructions
    style_instructions = {
        "default": "Use a neutral, informative tone.",
        "academic": "Use formal, academic language appropriate for scholarly contexts.",
        "simple": "Use simple, straightforward language accessible to all readers.",
        "bullet": "Format the summary as bullet points highlighting key information."
    }
    
    prompt = f"""Summarize the following text:

{text}

{length_instructions.get(length, length_instructions['medium'])}
{style_instructions.get(style, style_instructions['default'])}

Focus on the most important information and key points while maintaining accuracy.
"""
    return prompt

def summarize_with_anthropic(text, length, style):
    """Generate summary using Anthropic's Claude API"""
    
    prompt = generate_prompt(text, length, style)
    
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Extract the summary from the response
        if "content" in result and len(result["content"]) > 0:
            for content_block in result["content"]:
                if content_block["type"] == "text":
                    return content_block["text"]
        
        # Fallback if the response format is unexpected
        return "The summary could not be generated. Please try again."
        
    except Exception as e:
        logger.error(f"Anthropic API error: {str(e)}")
        # Fallback to OpenAI if Anthropic fails
        if OPENAI_API_KEY:
            logger.info("Falling back to OpenAI")
            return summarize_with_openai(text, length, style)
        raise

def summarize_with_openai(text, length, style):
    """Generate summary using OpenAI's API"""
    
    prompt = generate_prompt(text, length, style)
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5,
                "max_tokens": 1024
            }
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Extract the summary
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
            
        # Fallback
        return "The summary could not be generated. Please try again."
        
    except Exception as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise

# Add route to serve the frontend
@app.route('/', methods=['GET'])
def serve_frontend():
    return send_from_directory('.', 'index.html')

# Add a mock route for testing without API keys
@app.route('/mock-summarize', methods=['POST'])
def mock_summarize():
    try:
        data = request.json
        text = data.get('text', '')
        
        # Very simple mock summarization
        sentences = text.split('.')
        if len(sentences) <= 3:
            summary = text
        else:
            summary = '. '.join(sentences[:3]) + '.'
            
        # Artificial delay to simulate API call
        time.sleep(1)
        
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Check if API keys are available
    if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
        logger.warning("No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env file.")
        logger.info("You can still use the app with basic functionality via the mock API.")
    
    try:
        port = int(os.environ.get("PORT", 5000))
        print(f"Server started! Open http://localhost:{port} in your browser")
        app.run(debug=True, host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

# Built by Kshitiz Singh