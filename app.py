from flask import Flask, request, jsonify, render_template
import requests
import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)

# Securely load Hugging Face API key and model URL
HUGGINGFACE_API_KEY = ("")
HUGGINGFACE_MODEL_URL = os.getenv("HUGGINGFACE_MODEL_URL", "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct")

# Set up logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/generate', methods=['POST'])
def generate():
    # Get form data
    category = request.form.get('category')
    reason = request.form.get('reason')
    
    # Basic input validation
    if not category or not reason:
        return jsonify({'excuse': 'Please provide both category and reason.'}), 400
    
    prompt = f"Write a formal excuse letter for missing {category} because {reason}."

    headers = {
        "Authorization": f"Bearer {HUGGINGFACE_API_KEY}"
    }

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 150}
    }

    try:
        # Send request to Hugging Face API
        logging.info(f"Sending request to Hugging Face API with prompt: {prompt}")
        response = requests.post(HUGGINGFACE_MODEL_URL, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the response code is 4xx/5xx
        
        result = response.json()

        if isinstance(result, list) and 'generated_text' in result[0]:
            return jsonify({'excuse': result[0]['generated_text']})
        elif isinstance(result, dict) and 'generated_text' in result:
            return jsonify({'excuse': result['generated_text']})
        elif 'error' in result:
            return jsonify({'excuse': f"API Error: {result['error']}"}), 500
        else:
            return jsonify({'excuse': "Sorry, the model did not return a valid response."}), 500
    except requests.exceptions.RequestException as e:
        logging.error(f"Error while making the API request: {str(e)}")
        return jsonify({'excuse': f"Error: {str(e)}"}), 500
    except Exception as e:
        logging.error(f"General error: {str(e)}")
        return jsonify({'excuse': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
