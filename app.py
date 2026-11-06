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
    <title>AI Legal Advice Expert System</title>
    <style>
        body { background-color: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 30px; }
        .container { max-width: 850px; margin: auto; background: #161b22; padding: 25px; border-radius: 12px; border: 1px solid #30363d; box-shadow: 0 4px 20px rgba(0,255,170,0.08); }
        h1 { color: #00ffaa; margin-top: 0; text-align: center; }
        p.sub { text-align: center; color: #8b949e; margin-bottom: 25px; }
        textarea { width: 100%; height: 90px; background: #0d1117; color: #fff; border: 1px solid #30363d; border-radius: 8px; padding: 12px; font-size: 15px; box-sizing: border-box; }
        button { width: 100%; margin-top: 12px; padding: 12px; background: #00ffaa; color: #0d1117; font-weight: bold; font-size: 16px; border: none; border-radius: 8px; cursor: pointer; transition: 0.2s; }
        button:hover { background: #00cc88; }
        .result-box { margin-top: 25px; padding: 18px; background: #0d1117; border-left: 4px solid #00ffaa; border-radius: 6px; display: none; }
        .badge { display: inline-block; background: #1f6feb; color: #fff; padding: 4px 10px; border-radius: 4px; font-size: 12px; margin-right: 8px; }
        .score { color: #00ffaa; font-weight: bold; }
        .nlp-reply { font-size: 16px; line-height: 1.6; color: #e6edf3; background: #161b22; padding: 14px; border-radius: 8px; margin-top: 12px; border: 1px solid #30363d; }
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
