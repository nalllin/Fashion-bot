"""Demo web UI for interacting with the AIChatStylist.

This lightweight Flask application provides a minimal chat interface for the
stylist along with support for uploading outfit images and generating new
images. It is intended solely as a demonstration and makes no attempt to
handle authentication, persistence, or styling beyond the basics.

To run this app locally, install Flask (`pip install flask`) and run:

    python demo_ui.py

It will start a web server on http://localhost:5000. Open that URL in
your browser to interact with the stylist. You can send text messages,
upload an outfit image to describe, and request image generation by
entering a prompt in the "Generate Image" field.

NOTE: Image generation requires a valid OPENAI_API_KEY environment
variable to be set. If unset, the server will return an error when
attempting to generate an image.
"""

from __future__ import annotations

import os
import tempfile
from flask import Flask, request, render_template_string, jsonify

from chatbot import AIChatStylist

# Create the Flask app and a single stylist instance.
app = Flask(__name__)
stylist = AIChatStylist()


@app.route("/", methods=["GET"])
def index() -> str:
    """Serve a very simple chat page."""
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>AI Fashion Stylist Demo</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f7f7f7;
            }
            #chat {
                max-width: 800px;
                margin: 0 auto;
                padding: 1rem;
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            #messages {
                height: 300px;
                overflow-y: auto;
                border: 1px solid #ddd;
                padding: 0.5rem;
                margin-bottom: 1rem;
                background-color: #fafafa;
            }
            .message {
                margin-bottom: 0.5rem;
            }
            .user {
                color: #1a73e8;
                font-weight: bold;
            }
            .bot {
                color: #d61a3c;
                font-weight: bold;
            }
            #input-area {
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }
            input[type="text"] {
                padding: 0.5rem;
                border-radius: 4px;
                border: 1px solid #ccc;
                width: 100%;
            }
            input[type="file"] {
                margin-top: 0.25rem;
            }
            button {
                padding: 0.5rem 1rem;
                border: none;
                background-color: #1a73e8;
                color: #fff;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #1255a0;
            }
            .image-preview {
                max-width: 100%;
                margin-top: 0.5rem;
            }
        </style>
    </head>
    <body>
        <div id="chat">
            <h2>AI Fashion Stylist Demo</h2>
            <div id="messages"></div>
            <div id="input-area">
                <label for="message">Your message:</label>
                <input type="text" id="message" placeholder="Ask for style advice..." />
                <label for="image">Upload outfit image:</label>
                <input type="file" id="image" accept="image/*" />
                <label for="generate">Generate image (optional):</label>
                <input type="text" id="generate" placeholder="e.g., a chic summer outfit" />
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            const messagesDiv = document.getElementById('messages');

            function appendMessage(sender, text) {
                const div = document.createElement('div');
                div.className = 'message';
                div.innerHTML = `<span class="${sender}">${sender}:</span> ${text}`;
                messagesDiv.appendChild(div);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }

            async function sendMessage() {
                const messageInput = document.getElementById('message');
                const imageInput = document.getElementById('image');
                const generateInput = document.getElementById('generate');
                const message = messageInput.value.trim();
                const generateText = generateInput.value.trim();
                const formData = new FormData();
                if (message) {
                    formData.append('message', message);
                    appendMessage('You', message);
                }
                if (imageInput.files.length > 0) {
                    formData.append('image', imageInput.files[0]);
                }
                if (generateText) {
                    formData.append('generate_prompt', generateText);
                    appendMessage('You', '[generate image] ' + generateText);
                }
                // Reset inputs
                messageInput.value = '';
                generateInput.value = '';
                imageInput.value = '';
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    if (data.message_response) {
                        appendMessage('Stylist', data.message_response);
                    }
                    if (data.image_description) {
                        appendMessage('Stylist', '[Image description] ' + data.image_description);
                    }
                    if (data.generated_image_url) {
                        appendMessage('Stylist', '[Generated image]');
                        const img = document.createElement('img');
                        img.src = data.generated_image_url;
                        img.alt = 'Generated outfit';
                        img.className = 'image-preview';
                        messagesDiv.appendChild(img);
                        messagesDiv.scrollTop = messagesDiv.scrollHeight;
                    }
                } catch (err) {
                    appendMessage('Stylist', 'Error: ' + err);
                }
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)


@app.route("/chat", methods=["POST"])
def chat() -> tuple[any, int]:
    """Handle a chat request from the front-end.

    Expects form data with one or more of the following fields:
    - `message`: the user's text message.
    - `image`: a file upload containing an outfit image.
    - `generate_prompt`: a prompt for image generation.

    Returns a JSON object containing:
    - `message_response`: the stylist's text reply (if any).
    - `image_description`: the description generated for the uploaded outfit image (if any).
    - `generated_image_url`: the URL of the generated image (if requested).
    """
    response: dict[str, str] = {}

    # Process uploaded image
    if 'image' in request.files:
        file = request.files['image']
        if file and file.filename:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                file.save(tmp.name)
                try:
                    description = stylist.add_outfit_image(tmp.name)
                    response['image_description'] = description
                finally:
                    os.unlink(tmp.name)

    # Process text message
    message = request.form.get('message', '').strip()
    if message:
        try:
            reply = stylist.chat(message)
            response['message_response'] = reply
        except Exception as exc:
            response['message_response'] = f"Error: {exc}"

    # Process image generation request
    generate_prompt = request.form.get('generate_prompt', '').strip()
    if generate_prompt:
        try:
            url = stylist.generate_image(generate_prompt)
            response['generated_image_url'] = url
        except Exception as exc:
            response['generated_image_url'] = ''
            # Append error to message_response if present
            current = response.get('message_response', '')
            response['message_response'] = (current + f"\nImage generation error: {exc}").strip()

    return jsonify(response), 200


if __name__ == '__main__':
    # Expose the Flask development server
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
