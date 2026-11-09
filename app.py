from flask import Flask, request, render_template_string, jsonify
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

print("Loading Vector Database...")
with open("laws_vector_db.pkl", "rb") as f:
    df = pickle.load(f)

print("Loading Prebuilt NLP Models (Embedding + Conversational Generator)...")


_embedder = None
_nlp_chatbot = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder


HTML_UI = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Legal Advice Expert System</title>
    <style>
        :root {
            --bg-main: #090d16;
            --bg-card: rgba(22, 27, 34, 0.85);
            --bg-inner: #0d1117;
            --border-color: #30363d;
            --accent: #00ffaa;
            --accent-hover: #00cc88;
            --accent-glow: rgba(0, 255, 170, 0.18);
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
        }

        * {
            box-sizing: border-box;
        }

        body {
            background-color: var(--bg-main);
            background-image: 
                radial-gradient(circle at 15% 15%, rgba(0, 255, 170, 0.06) 0%, transparent 35%),
                radial-gradient(circle at 85% 85%, rgba(31, 111, 235, 0.07) 0%, transparent 35%);
            color: var(--text-primary);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 40px 20px;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .container {
            width: 100%;
            max-width: 820px;
            margin: auto;
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 36px;
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6), 0 0 30px rgba(0, 255, 170, 0.05);
            transition: all 0.3s ease;
        }

        h1 {
            color: #ffffff;
            background: linear-gradient(90deg, #ffffff, var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-top: 0;
            margin-bottom: 8px;
            text-align: center;
            font-size: clamp(1.6rem, 4vw, 2.25rem);
            font-weight: 800;
            letter-spacing: -0.5px;
        }

        p.sub {
            text-align: center;
            color: var(--text-secondary);
            margin-top: 0;
            margin-bottom: 28px;
            font-size: clamp(0.875rem, 2vw, 1rem);
            letter-spacing: 0.2px;
        }

        textarea {
            width: 100%;
            min-height: 120px;
            background: var(--bg-inner);
            color: var(--text-primary);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            font-size: 15px;
            line-height: 1.5;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.25s ease, box-shadow 0.25s ease;
        }

        textarea::placeholder {
            color: #586069;
        }

        textarea:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        button {
            width: 100%;
            margin-top: 16px;
            padding: 15px 24px;
            background: linear-gradient(135deg, #00ffaa 0%, #00d68f 100%);
            color: #06130e;
            font-weight: 700;
            font-size: 16px;
            letter-spacing: 0.3px;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease, filter 0.2s ease;
            box-shadow: 0 4px 15px rgba(0, 255, 170, 0.25);
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 255, 170, 0.35);
            filter: brightness(1.05);
        }

        button:active {
            transform: translateY(0);
        }

        .result-box {
            margin-top: 30px;
            padding: 24px;
            background: var(--bg-inner);
            border: 1px solid var(--border-color);
            border-left: 4px solid var(--accent);
            border-radius: 12px;
            display: none;
            animation: slideFadeIn 0.35s ease-out forwards;
        }

        @keyframes slideFadeIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .result-box > div:first-child {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 10px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(48, 54, 61, 0.6);
            margin-bottom: 16px;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            background: rgba(31, 111, 235, 0.15);
            color: #58a6ff;
            border: 1px solid rgba(88, 166, 255, 0.3);
            padding: 6px 12px;
            border-radius: 999px;
            font-size: 12.5px;
            font-weight: 600;
            letter-spacing: 0.3px;
            margin-right: 0;
        }

        #secBadge {
            background: rgba(137, 87, 229, 0.15) !important;
            color: #d2a8ff !important;
            border: 1px solid rgba(188, 140, 255, 0.3) !important;
        }

        .score {
            color: var(--accent);
            font-weight: 700;
            font-size: 13px;
            margin-left: auto;
            background: rgba(0, 255, 170, 0.1);
            border: 1px solid rgba(0, 255, 170, 0.25);
            padding: 6px 12px;
            border-radius: 999px;
        }

        .result-box h3,
        .result-box h4 {
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-top: 20px;
            margin-bottom: 8px;
        }

        .nlp-reply {
            font-size: 15.5px;
            line-height: 1.7;
            color: var(--text-primary);
            background: #161b22;
            padding: 18px;
            border-radius: 10px;
            margin-top: 10px;
            border: 1px solid var(--border-color);
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
        }

        #rawLaw {
            background: rgba(255, 255, 255, 0.02);
            padding: 12px 14px;
            border-radius: 8px;
            border-left: 3px solid #8b949e;
            line-height: 1.6;
            font-size: 14.5px;
        }

        #actionStep {
            background: rgba(63, 185, 80, 0.08);
            border: 1px solid rgba(63, 185, 80, 0.25);
            color: #e6edf3;
            padding: 12px 14px;
            border-radius: 8px;
            line-height: 1.6;
            font-size: 15px;
        }

        /* Responsive Mobile Layout */
        @media (max-width: 600px) {
            body {
                padding: 16px 12px;
                align-items: flex-start;
            }
            .container {
                padding: 22px 18px;
                border-radius: 14px;
            }
            .result-box {
                padding: 18px 14px;
            }
            .score {
                margin-left: 0;
            }
            button {
                padding: 14px 18px;
                font-size: 15px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>⚖️ AI Legal Expert System</h1>
        <p class="sub">Vector Similarity Search + Prebuilt NLP Interactive Advisor</p>
        <textarea id="userInput" placeholder="Describe your problem (e.g., 'I bought a phone online and it arrived broken, seller is refusing refund')..."></textarea>
        <button onclick="askExpert()">Analyze Problem & Find Applicable Law</button>

        <div id="result" class="result-box">
            <div>
                <span class="badge" id="catBadge"></span>
                <span class="badge" id="secBadge" style="background:#8957e5;"></span>
                <span class="score" id="simScore"></span>
            </div>
            <h3 style="margin-bottom:5px; color:#58a6ff;">🤖 Interactive NLP Advisor Response:</h3>
            <div class="nlp-reply" id="nlpReply"></div>
            <h4 style="margin-bottom:5px; color:#8b949e;">📜 Exact Law Matched from Vector Column:</h4>
            <p id="rawLaw" style="color:#c9d1d9; font-style:italic; margin-top:4px;"></p>
            <h4 style="margin-bottom:5px; color:#3fb950;">✅ Recommended Next Step:</h4>
            <p id="actionStep" style="margin-top:4px;"></p>
        </div>
    </div>

    <script>
        async function askExpert() {
            const query = document.getElementById("userInput").value;
            if (!query.trim()) return;
            const btn = document.querySelector("button");
            btn.innerText = "Converting to Vector & Running NLP...";
            
            const res = await fetch("/analyze", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({query: query})
            });
            const data = await res.json();
            
            document.getElementById("catBadge").innerText = data.category;
            document.getElementById("secBadge").innerText = data.section;
            document.getElementById("simScore").innerText = "Vector Match: " + data.similarity + "%";
            document.getElementById("nlpReply").innerText = data.chatbot_response;
            document.getElementById("rawLaw").innerText = data.law_text;
            document.getElementById("actionStep").innerText = data.action_steps;
            
            document.getElementById("result").style.display = "block";
            btn.innerText = "Analyze Problem & Find Applicable Law";
        }
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_UI)

@app.route("/analyze", methods=["POST"])
def analyze():
    user_query = request.json.get("query", "")
    
    # 1. Convert user input into a vector
    user_vector = get_embedder().encode([user_query], convert_to_numpy=True)
    
    # 2. Compare user vector with the 'Vector' column in our DataFrame
    law_vectors = np.vstack(df['Vector'].values)
    similarities = cosine_similarity(user_vector, law_vectors)[0]
    
    # 3. Find the closest vector match
    best_idx = int(np.argmax(similarities))
    best_score = round(float(similarities[best_idx]) * 100, 2)
    matched_row = df.iloc[best_idx]
    
    # 4. Use Prebuilt NLP Model to interact with the user using the matched law
    nlp_output = f"Based on your description, this falls under {matched_row['Section']}. The law states: {matched_row['Law_Text']} We highly recommend you take the following action: {matched_row['Action_Steps']}"
    
    return jsonify({
        "category": matched_row['Category'],
        "section": matched_row['Section'],
        "law_text": matched_row['Law_Text'],
        "action_steps": matched_row['Action_Steps'],
        "similarity": best_score,
        "chatbot_response": nlp_output
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
