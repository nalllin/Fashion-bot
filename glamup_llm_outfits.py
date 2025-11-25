"""GlamUp LLM outfits UI with questionnaire and outfit card preview."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template_string, request

from chatbot import AIChatStylist

app = Flask(__name__)
stylist = AIChatStylist()

QUESTIONNAIRE: Dict[str, List[str]] = {
    "Style Identity Questions": [
        "How would you describe your personal style in 3 words? (e.g., casual, edgy, classic)",
        "If your wardrobe had a \"mood,\" what would it be? (playful, minimal, bold, clean, boho, etc.)",
    ],
    "Clothing Preferences": [
        "What kind of outfits make you feel most confident?",
        "Do you prefer more: timeless basics or trendy statement pieces?",
        "Do you like fitted silhouettes, relaxed fits, or a mix?",
        "What's your go-to outfit for: A workout / A casual outing / A night out",
    ],
    "Color & Print Choices": [
        "Which colors do you gravitate most toward?",
        "Any colors you avoid completely?",
        "Do you prefer solid colors, minimal prints, bold prints?",
    ],
    "Shopping Behavior": [
        "When shopping, what do you prioritize first: comfort, trendiness, budget, or brand?",
        "Do you shop because you need something specific or for everyday wear?",
        "How often do you refresh your wardrobe? (monthly, seasonally, rarely)",
    ],
}


def build_questionnaire_html() -> str:
    sections = []
    for title, questions in QUESTIONNAIRE.items():
        items = "".join(f"<li>{q}</li>" for q in questions)
        sections.append(f"<div class='q-section'><h4>{title}</h4><ul>{items}</ul></div>")
    return "".join(sections)


@app.route("/", methods=["GET"])
def index() -> str:
    questionnaire_html = build_questionnaire_html()
    html_template = f"""
    <!DOCTYPE html>
    <html lang='en'>
    <head>
        <meta charset='UTF-8' />
        <meta name='viewport' content='width=device-width, initial-scale=1.0' />
        <title>GlamUp Stylist Console</title>
        <style>
            :root {{
                --bg: #0f1117;
                --panel: #161a23;
                --accent: #ff8fb1;
                --text: #f7f7f7;
                --muted: #a5a7b0;
                --border: #2b3040;
            }}
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                font-family: 'Inter', Arial, sans-serif;
                background: linear-gradient(135deg, #0f1117, #1a2030);
                color: var(--text);
                min-height: 100vh;
            }}
            header {{
                padding: 20px 28px;
                display: flex;
                align-items: baseline;
                justify-content: space-between;
            }}
            header h1 {{ margin: 0; font-size: 28px; letter-spacing: 0.5px; }}
            header span {{ color: var(--muted); font-size: 14px; }}
            main {{ display: grid; grid-template-columns: 2fr 1.5fr; gap: 18px; padding: 0 24px 24px; }}
            .panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 18px; box-shadow: 0 16px 50px rgba(0,0,0,0.35); }}
            .panel h3 {{ margin-top: 0; letter-spacing: 0.3px; }}
            .chat-window {{ display: flex; flex-direction: column; gap: 12px; height: 70vh; }}
            #messages {{ flex: 1; overflow-y: auto; padding: 12px; border: 1px solid var(--border); border-radius: 12px; background: #0f1117; }}
            .message {{ margin-bottom: 10px; }}
            .message span {{ display: block; font-size: 13px; color: var(--muted); }}
            .bubble {{ display: inline-block; padding: 10px 12px; border-radius: 12px; margin-top: 4px; max-width: 90%; line-height: 1.4; }}
            .you {{ background: #1f2433; color: var(--text); }}
            .bot {{ background: rgba(255, 143, 177, 0.08); color: #ffd6e6; border: 1px solid rgba(255, 143, 177, 0.3); }}
            .inputs {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px; }}
            .inputs label {{ font-size: 12px; color: var(--muted); margin-bottom: 4px; display: block; }}
            input[type='text'], input[type='file'] {{ width: 100%; padding: 10px; border-radius: 10px; border: 1px solid var(--border); background: #0f1117; color: var(--text); }}
            button {{ padding: 10px 12px; border-radius: 10px; border: none; background: linear-gradient(120deg, #ff8fb1, #ff6c9f); color: #0f1117; font-weight: 700; cursor: pointer; box-shadow: 0 8px 20px rgba(255,108,159,0.35); }}
            button:hover {{ filter: brightness(1.05); }}
            .card {{ display: grid; grid-template-columns: 180px 1fr; gap: 16px; align-items: start; }}
            .card .preview {{ position: relative; border: 1px dashed var(--border); border-radius: 12px; padding: 12px; text-align: center; background: #0f1117; min-height: 380px; }}
            .card .preview img {{ max-width: 100%; border-radius: 10px; box-shadow: 0 20px 50px rgba(0,0,0,0.45); }}
            .thumbs {{ display: flex; flex-direction: column; gap: 10px; }}
            .thumbs img {{ width: 100%; border-radius: 10px; border: 1px solid var(--border); background: #0f1117; padding: 6px; }}
            .q-section {{ margin-bottom: 12px; }}
            .q-section h4 {{ margin: 0 0 6px; color: #ffd6e6; }}
            .q-section ul {{ margin: 0; padding-left: 18px; color: var(--muted); line-height: 1.5; }}
        </style>
    </head>
    <body>
        <header>
            <div>
                <h1>GlamUp Stylist</h1>
                <span>Curated looks, questionnaire-driven insights, and outfit cards.</span>
            </div>
        </header>
        <main>
            <section class='panel chat-window'>
                <div>
                    <h3>Chat with your stylist</h3>
                    <div class='inputs'>
                        <div>
                            <label for='message'>Message</label>
                            <input type='text' id='message' placeholder='Tell us what you need...' />
                        </div>
                        <div>
                            <label for='image'>Upload base/model image</label>
                            <input type='file' id='image' accept='image/*' />
                        </div>
                        <div>
                            <label for='items'>Product thumbnails (for card)</label>
                            <input type='file' id='items' accept='image/*' multiple />
                        </div>
                        <div>
                            <label for='generate'>Prompt to generate a new look</label>
                            <input type='text' id='generate' placeholder='e.g., neon athleisure with metallic accents' />
                        </div>
                    </div>
                </div>
                <div id='messages'></div>
                <div style='display:flex; gap: 10px; justify-content:flex-end;'>
                    <button onclick='sendMessage()'>Send to stylist</button>
                    <button onclick='clearChat()' style='background:#1f2433; color: var(--text); box-shadow:none;'>Clear</button>
                </div>
            </section>
            <section class='panel'>
                <h3>Outfit card preview</h3>
                <div class='card'>
                    <div class='thumbs' id='thumbs'></div>
                    <div class='preview'>
                        <img id='outfit-preview' alt='Outfit preview' />
                        <p id='preview-caption' style='color: var(--muted); margin-top: 10px;'>Generated or uploaded looks will appear here.</p>
                    </div>
                </div>
                <div style='margin-top:16px;'>
                    <h3>Style Identity Questionnaire</h3>
                    {questionnaire_html}
                </div>
            </section>
        </main>

        <script>
            const messagesDiv = document.getElementById('messages');
            const preview = document.getElementById('outfit-preview');
            const caption = document.getElementById('preview-caption');
            const thumbs = document.getElementById('thumbs');

            function appendMessage(sender, text) {{
                const container = document.createElement('div');
                container.className = 'message';
                const label = document.createElement('span');
                label.textContent = sender;
                const bubble = document.createElement('div');
                bubble.className = 'bubble ' + (sender === 'You' ? 'you' : 'bot');
                bubble.textContent = text;
                container.appendChild(label);
                container.appendChild(bubble);
                messagesDiv.appendChild(container);
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }}

            function clearChat() {{
                messagesDiv.innerHTML = '';
                preview.src = '';
                caption.textContent = 'Generated or uploaded looks will appear here.';
                thumbs.innerHTML = '';
            }}

            function handleThumbs(files) {{
                thumbs.innerHTML = '';
                Array.from(files).slice(0, 6).forEach((file) => {{
                    const reader = new FileReader();
                    reader.onload = (e) => {{
                        const img = document.createElement('img');
                        img.src = e.target.result;
                        img.alt = file.name;
                        thumbs.appendChild(img);
                    }};
                    reader.readAsDataURL(file);
                }});
            }}

            async function sendMessage() {{
                const messageInput = document.getElementById('message');
                const imageInput = document.getElementById('image');
                const generateInput = document.getElementById('generate');
                const itemInput = document.getElementById('items');

                const message = messageInput.value.trim();
                const generateText = generateInput.value.trim();
                const formData = new FormData();

                if (message) {{
                    formData.append('message', message);
                    appendMessage('You', message);
                }}

                if (imageInput.files.length > 0) {{
                    formData.append('image', imageInput.files[0]);
                    const reader = new FileReader();
                    reader.onload = (e) => {{ preview.src = e.target.result; caption.textContent = 'Base/model image loaded.'; }};
                    reader.readAsDataURL(imageInput.files[0]);
                }}

                if (generateText) {{
                    formData.append('generate_prompt', generateText);
                    appendMessage('You', '[generate image] ' + generateText);
                }}

                if (itemInput.files.length > 0) {{
                    handleThumbs(itemInput.files);
                }}

                try {{
                    const response = await fetch('/chat', {{ method: 'POST', body: formData }});
                    const data = await response.json();
                    if (data.message_response) {{ appendMessage('Stylist', data.message_response); }}
                    if (data.image_description) {{ appendMessage('Stylist', '[Image description] ' + data.image_description); }}
                    if (data.generated_image_url) {{
                        preview.src = data.generated_image_url;
                        caption.textContent = 'Latest generated look';
                    }}
                }} catch (err) {{
                    appendMessage('Stylist', 'Error: ' + err);
                }} finally {{
                    messageInput.value = '';
                    generateInput.value = '';
                    imageInput.value = '';
                }}
            }}
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)


@app.route("/chat", methods=["POST"])
def chat() -> tuple[Any, int]:
    response: dict[str, str] = {}

    if "image" in request.files:
        file = request.files["image"]
        if file and file.filename:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                file.save(tmp.name)
                try:
                    description = stylist.add_outfit_image(tmp.name)
                    response["image_description"] = description
                finally:
                    os.unlink(tmp.name)

    message = request.form.get("message", "").strip()
    if message:
        try:
            response["message_response"] = stylist.chat(message)
        except Exception as exc:  # pragma: no cover - defensive
            response["message_response"] = f"Error: {exc}"

    generate_prompt = request.form.get("generate_prompt", "").strip()
    if generate_prompt:
        try:
            url = stylist.generate_image(generate_prompt)
            response["generated_image_url"] = url
        except Exception as exc:  # pragma: no cover - defensive
            response["generated_image_url"] = ""
            current = response.get("message_response", "")
            response["message_response"] = (current + f"\nImage generation error: {exc}").strip()

    return jsonify(response), 200


if __name__ == "__main__":  # pragma: no cover - manual launch
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=True)
