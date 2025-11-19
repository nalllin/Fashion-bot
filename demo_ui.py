# Demo web UI for Alle – AI Fashion Stylist
# - Upload image -> vision description
# - Normal chat -> text outfit plan + optional SDXL visual
# - "generate image : ..." -> direct SDXL image generation

from __future__ import annotations
import os
import tempfile
from flask import Flask, request, jsonify, render_template_string

from ai_chat_stylist.chatbot import AIChatStylist

app = Flask(__name__)
stylist = AIChatStylist()


# ---------- Helpers ----------

def _normalize_sdxl_output(obj) -> str:
    """Turn SDXL / Replicate output into a plain URL string."""
    url = obj
    if hasattr(url, "url"):
        url = url.url
    if isinstance(url, (list, tuple)) and url:
        url = url[0]
    if not url:
        return ""
    return str(url)


# ---------- UI ----------

@app.route("/", methods=["GET"])
def index():
    html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Alle – AI Fashion Stylist</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<style>
body {
  margin: 0;
  background: radial-gradient(circle at top, #020617 0, #020617 40%, #000 100%);
  font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
  color: #f9fafb;
}
.page { max-width: 900px; margin: 0 auto; padding: 20px; }
.card {
  background: #0b1220;
  border-radius: 16px;
  border: 1px solid rgba(148,163,184,0.35);
  box-shadow: 0 18px 45px rgba(15,23,42,0.85);
  display: flex; flex-direction: column;
  height: 90vh;
}

/* Header */
.header { padding: 14px 18px; border-bottom: 1px solid rgba(148,163,184,0.25); }
.header-title { font-size: 18px; font-weight: 600; }
.header-sub { font-size: 12px; color: #9ca3af; }

/* Messages */
.messages { flex: 1; overflow-y: auto; padding: 16px; }
.message-row { display: flex; margin-bottom: 14px; }
.message-row.user { justify-content: flex-end; }
.message-row.alle { justify-content: flex-start; }

.bubble {
  padding: 10px 14px;
  border-radius: 14px;
  max-width: 75%;
  font-size: 14px;
  line-height: 1.45;
  white-space: pre-wrap;
}
.bubble.user {
  background: linear-gradient(135deg, #22d3ee, #6366f1);
  color: #ecfeff;
  border-bottom-right-radius: 4px;
  box-shadow: 0 0 18px rgba(56,189,248,0.55);
}
.bubble.alle {
  background: #020617;
  color: #e5e7eb;
  border: 1px solid rgba(148,163,184,0.45);
  border-bottom-left-radius: 4px;
}

/* Images */
.chat-img {
  margin-top: 8px;
  max-width: 100%;
  border-radius: 12px;
  border: 1px solid rgba(148,163,184,0.7);
}

/* Footer */
.footer {
  padding: 12px 14px;
  border-top: 1px solid rgba(148,163,184,0.35);
  display: flex; gap: 10px; align-items: center;
}
textarea {
  flex: 1; height: 50px; resize: none;
  border-radius: 12px;
  border: 1px solid rgba(148,163,184,0.6);
  background: #020617; color: #f9fafb;
  padding: 10px 11px; font-size: 14px;
}
textarea::placeholder { color: #64748b; }

input[type="file"] { font-size: 11px; color: #9ca3af; }

button {
  min-width: 80px;
  background: radial-gradient(circle at 0 0, #22d3ee 0, #6366f1 50%, #0ea5e9 100%);
  border: none; border-radius: 999px;
  padding: 10px 16px; color: #f9fafb; font-size: 14px;
  cursor: pointer;
  box-shadow: 0 0 18px rgba(56,189,248,0.55);
}
button:hover { transform: translateY(-1px); }
button:disabled { opacity: 0.5; cursor: default; }

/* Inline button inside chat */
.inline-btn {
  min-width: 0;
  padding: 6px 10px;
  font-size: 13px;
  margin-top: 6px;
}
</style>
</head>

<body>
<div class="page">
  <div class="card">

    <div class="header">
      <div class="header-title">Alle — AI Fashion Stylist</div>
      <div class="header-sub">
        Get outfit ideas, upload looks, and generate AI visuals.
        You can also type: <strong>generate image : your outfit description</strong>.
      </div>
    </div>

    <div class="messages" id="messages">
      <div class="message-row alle">
        <div class="bubble alle">
Hi, I'm Alle 👗

Tell me the occasion or vibe.  
I'll give a text outfit plan first.  
Upload a clothing image for feedback, or say:

generate image : light denim jacket, black turtleneck, beige wide-leg pants, ankle boots, crossbody bag
        </div>
      </div>
    </div>

    <div class="footer">
      <textarea id="chat_message" placeholder="Ask Alle anything…"></textarea>
      <input type="file" id="chat_image" accept="image/*">
      <button id="chat_btn" onclick="sendChat()">Send</button>
    </div>

  </div>
</div>

<script>
const messagesBox = document.getElementById("messages");
let pendingVisual = null;

function appendMessage(sender, text, img=null) {
  const row = document.createElement("div");
  row.className = "message-row " + sender;

  const bubble = document.createElement("div");
  bubble.className = "bubble " + sender;

  if (text) bubble.textContent = text;

  if (img) {
    const tag = document.createElement("img");
    tag.src = img;
    tag.className = "chat-img";
    bubble.appendChild(tag);
  }

  row.appendChild(bubble);
  messagesBox.appendChild(row);
  messagesBox.scrollTop = messagesBox.scrollHeight;
}

function appendVisualOffer() {
  const row = document.createElement("div");
  row.className = "message-row alle";

  const bubble = document.createElement("div");
  bubble.className = "bubble alle";

  const text = document.createElement("div");
  text.textContent = "Would you like to see a generated outfit for this plan?";

  const btn = document.createElement("button");
  btn.textContent = "Show generated outfit";
  btn.className = "inline-btn";
  btn.onclick = function() { showPendingVisual(row); };

  bubble.appendChild(text);
  bubble.appendChild(btn);
  row.appendChild(bubble);
  messagesBox.appendChild(row);
  messagesBox.scrollTop = messagesBox.scrollHeight;
}

function showPendingVisual(row) {
  if (!pendingVisual) return;
  appendMessage("alle", "Here’s a rough generated outfit ✨", pendingVisual);
  pendingVisual = null;
  row.remove();
}

async function sendChat() {
  const msgEl = document.getElementById("chat_message");
  const imgEl = document.getElementById("chat_image");
  const btn = document.getElementById("chat_btn");

  const message = msgEl.value.trim();
  const fd = new FormData();

  if (!message && imgEl.files.length === 0) return;

  if (message) {
    fd.append("message", message);
    appendMessage("user", message);
  }

  if (imgEl.files.length > 0) {
    const file = imgEl.files[0];
    const preview = URL.createObjectURL(file);
    // just preview, no "[Image uploaded]" text
    appendMessage("user", "", preview);
    fd.append("image", file);
  }

  msgEl.value = "";
  imgEl.value = "";
  btn.disabled = true;

  try {
    const res = await fetch("/chat", { method: "POST", body: fd });
    const data = await res.json();

    if (data.reply) appendMessage("alle", data.reply);
    if (data.image_description) appendMessage("alle", data.image_description);

    // Direct SDXL for "generate image : ..."
    if (data.generated_image_url) {
      appendMessage(
        "alle",
        "Here’s the generated outfit based on your description ✨",
        data.generated_image_url
      );
    }

    // Optional SDXL for normal outfit questions
    if (data.offer_visual && data.visual_url) {
      pendingVisual = data.visual_url;
      appendVisualOffer();
    }

  } catch (err) {
    appendMessage("alle", "Error: " + err);
  } finally {
    btn.disabled = false;
  }
}
</script>

</body>
</html>
"""
    return render_template_string(html)


# ---------- API: Chat + SDXL + Image Upload ----------

@app.route("/chat", methods=["POST"])
def chat():
    message = (request.form.get("message") or "").strip()
    file = request.files.get("image")

    # ----- DIRECT SDXL GENERATION -----
    if message.lower().startswith("generate image"):
        prompt = message.split(":", 1)[-1].strip() or message
        try:
            summary, _card, img = stylist.glamup_outfit_with_image(prompt)
            gen_url = _normalize_sdxl_output(img)
            return jsonify({
                "reply": "Here’s the generated outfit based on your description ✨",
                "generated_image_url": gen_url
            })
        except Exception as exc:
            return jsonify({"reply": f"Couldn’t generate image: {exc}"})


    # ----- NORMAL CHAT -----
    out: dict = {}

    # Image uploaded → save to temp file, pass path to stylist
    if file and file.filename:
        suffix = os.path.splitext(file.filename)[1] or ".png"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            file.save(tmp.name)
            desc = stylist.add_outfit_image(tmp.name)
            out["image_description"] = desc
        except Exception as exc:
            out["image_description"] = f"Sorry, I couldn't read the image: {exc}"
        finally:
            tmp.close()
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    # Text reply
    if message:
        try:
            r = stylist.chat(message)
        except Exception as exc:
            r = f"Error: {exc}"
        out["reply"] = r

        # Outfit-trigger → SDXL (optional)
        lower = message.lower()
        trigger = any(k in lower for k in [
            "outfit", "style me", "what should i wear", "dress for", "wear to"
        ])

        if trigger:
            try:
                summary, _card, img = stylist.glamup_outfit_with_image(message)
                if summary:
                    out["reply"] = (out["reply"] + "\n\n" + summary).strip()
                gen_url = _normalize_sdxl_output(img)
                if gen_url:
                    out["offer_visual"] = True
                    out["visual_url"] = gen_url
            except Exception:
                pass

    return jsonify(out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
