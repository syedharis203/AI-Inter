from flask import Flask, request, render_template, jsonify, session, redirect
import os
import pdfplumber
import requests
import json
import pinecone
import random

# --- Flask Setup ---
app = Flask(__name__)
app.secret_key = "super-secret-key"
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Pinecone Setup ---
pinecone_index = None
try:
pinecone.init(api_key="pcsk_2CCJh1_KBatSJodZV1MYdVvYokH4WpHfi2BHiSKXCHhSz76XWiwi64Qvau7SWZx6JxJEqJ", environment="us-east-1-aws")
index_name = "rag"


    if index_name not in pinecone.list_indexes():
        print(f"[INFO] Creating Pinecone index '{index_name}' with 1024 dimensions.")
        pinecone.create_index(name=index_name, dimension=1024, metric="cosine")

    pinecone_index = pinecone.Index(index_name)
    print(f"[INFO] Pinecone index '{index_name}' initialized.")

except Exception as e:
    print(f"[WARNING] Pinecone initialization failed: {e}")
    pinecone_index = None

# --- Ollama Setup ---
OLLAMA_BASE_URL = "https://ai.thecodehub.digital"
OLLAMA_EMBED_MODEL = "bge-m3:latest"
OLLAMA_CHAT_MODEL = "mistral-small3.1:latest"

# --- Helpers ---
def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as e:
                print(f"[PDF warning] page {page.page_number}:", e)
    return text

def ollama_chat(prompt):
    try:
        res = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={"model": OLLAMA_CHAT_MODEL, "messages": [{"role": "user", "content": prompt}]},
            stream=True
        )
        full_response = ""
        for line in res.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "message" in data and "content" in data["message"]:
                        full_response += data["message"]["content"]
                except json.JSONDecodeError:
                    continue
        return full_response.strip() or "[Ollama chat error: Empty response]"
    except Exception as e:
        return f"[Ollama chat error: {str(e)}]"

def embed_text(text):
    try:
        res = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text}
        )
        return res.json()['embedding']
    except Exception as e:
        print(f"[Embedding error] {e}")
        return [0.0] * 1024

def extract_skills_with_ollama(resume_text, job_title):
    prompt = f"""
    Analyze this resume and extract the candidate's key technical skills.
    Group them into categories like Frontend, Backend, DevOps, Data, AI/ML, Other.

    Resume:\n{resume_text}
    Job Title (if any): {job_title}

    Output format:
    {{
        "Frontend": [...],
        "Backend": [...],
        "DevOps": [...],
        "Data": [...],
        "AI/ML": [...],
        "Other": [...]
    }}
    """
    try:
        response = ollama_chat(prompt)
        cleaned = response.strip().strip("` ")
        if cleaned.startswith("json"):
            cleaned = cleaned.replace("json", "").strip()
        return json.loads(cleaned)
    except Exception as e:
        return {"error": str(e)}

def generate_question_from_skill(skill):
    embed = embed_text(skill)
    context_text = ""

    if pinecone_index:
        try:
            pinecone_results = pinecone_index.query(vector=embed, top_k=5, include_metadata=True, namespace="interview")
            context_chunks = [m['metadata']['text'] for m in pinecone_results['matches'] if 'text' in m['metadata']]
            context_text = "\n\n".join(context_chunks)
        except Exception as e:
            print(f"[Pinecone query error] {e}")

    prompt = f"""
You're an AI interviewer. Ask a single friendly and relevant technical question based on the candidate's experience with {skill}.

Do not introduce the skill in the question. Do not mention the candidate’s name or background. Do not include explanations or summaries.
Use a friendly tone, like starting with 'Great!' or 'Thanks!'.

Use this reference material for ideas:
{context_text}

Only return the question string. No markdown or JSON.
    """
    return ollama_chat(prompt)

def get_initial_greeting():
    prompt = "You're an HR interviewer. Greet the candidate warmly and ask them to introduce themselves."
    return ollama_chat(prompt)

def evaluate_answer(answer_text):
    eval_prompt = f"""
    Evaluate if the following answer was written by an AI or a human.
    Return a JSON like: {{"score": 85, "label": "AI-like"}}.
    Answer:\n{answer_text}
    """
    try:
        response = ollama_chat(eval_prompt)
        cleaned = response.strip().strip("` ")
        if cleaned.startswith("json"):
            cleaned = cleaned.replace("json", "").strip()
        return json.loads(cleaned)
    except Exception as e:
        return {"score": 0, "label": "Unknown"}

def calculate_summary_score(transcript):
    if not transcript:
        return {"avg_score": 0, "ai_count": 0, "human_count": 0}
    
    total_score = 0
    ai_like = 0
    human_like = 0
    for entry in transcript:
        eval = entry.get("evaluation", {})
        score = eval.get("score", 0)
        label = eval.get("label", "").lower()
        total_score += score
        if "ai" in label:
            ai_like += 1
        elif "human" in label:
            human_like += 1

    avg = round(total_score / len(transcript), 1)
    return {"avg_score": avg, "ai_count": ai_like, "human_count": human_like}

# --- Routes ---
@app.route('/')
def home():
    return render_template('combined.html')

@app.route('/upload', methods=['POST'])
def upload_resume():
    file = request.files['resume']
    job_title = request.form.get("job_title", "").strip()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(filepath)
    else:
        text = open(filepath, 'r', encoding='utf-8').read()

    embedding = embed_text(text)

    if pinecone_index:
        try:
            pinecone_index.upsert(
                vectors=[{
                    "id": file.filename,
                    "values": embedding,
                    "metadata": {"text": text, "source": "resume", "job_title": job_title}
                }],
                namespace="interview"
            )
        except Exception as e:
            print(f"[Pinecone upsert error] {e}")
    else:
        print("[INFO] Pinecone index not initialized. Skipping upsert.")

    extracted_skills = extract_skills_with_ollama(text, job_title)
    session['skills'] = extracted_skills
    session['transcript'] = []
    session['asked_skills'] = []
    session['intro_done'] = False
    session['question_count'] = 0
    return jsonify({"status": "ok"})

@app.route('/next_question', methods=['POST'])
def next_question():
    data = request.json
    user_answer = data.get("answer", "")
    transcript = session.get('transcript', [])
    asked_skills = session.get('asked_skills', [])
    skill_dict = session.get('skills', {})
    intro_done = session.get('intro_done', False)
    question_count = session.get('question_count', 0)

    evaluation = None
    if 'last_question' in session and user_answer:
        evaluation = evaluate_answer(user_answer)
        transcript.append({
            "q": session['last_question'],
            "a": user_answer,
            "evaluation": evaluation
        })
        session['transcript'] = transcript

    if not intro_done:
        q = get_initial_greeting()
        session['last_question'] = q
        session['intro_done'] = True
        return jsonify({"done": False, "question": q})

    if question_count >= 5:
        summary = calculate_summary_score(session.get('transcript', []))
        return jsonify({
            "done": True,
            "message": "Thanks for taking the interview!",
            "summary": summary
        })

    flat_skills = [(cat, skill) for cat, lst in skill_dict.items() for skill in lst if skill not in asked_skills]
    if not flat_skills:
        summary = calculate_summary_score(session.get('transcript', []))
        return jsonify({
            "done": True,
            "message": "Thanks for taking the interview!",
            "summary": summary
        })

    cat, next_skill = random.choice(flat_skills)
    asked_skills.append(next_skill)
    session['asked_skills'] = asked_skills
    q = generate_question_from_skill(next_skill)
    session['last_question'] = q
    session['question_count'] = question_count + 1

    return jsonify({
        "done": False,
        "question": q,
        "evaluation": evaluation
    })

@app.route('/admin', methods=['GET', 'POST'])
def admin_panel():
    if request.method == 'POST':
        skill = request.form['skill']
        question = request.form['question']
        if pinecone_index:
            try:
                pinecone_index.upsert(
                    vectors=[{
                        "id": f"{skill}-{random.randint(1000,9999)}",
                        "values": embed_text(question),
                        "metadata": {"text": question, "source": "admin-upload"}
                    }],
                    namespace="interview"
                )
            except Exception as e:
                print(f"[Admin Pinecone upsert error] {e}")
        return redirect('/admin')

    return '''
    <h3>Admin Panel - Upload Question</h3>
    <form method="POST">
      Skill: <input type="text" name="skill"><br><br>
      Question:<br>
      <textarea name="question" rows="5" cols="50"></textarea><br><br>
      <input type="submit" value="Add">
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
