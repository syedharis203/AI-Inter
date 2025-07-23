AI Interview System (Resume-Based Chatbot)
This project is an AI-powered Interview Simulator that conducts interactive technical interviews based on a candidate's resume. It mimics an HR and technical interview, using your uploaded resume to drive questions through a RAG (Retrieval-Augmented Generation) system using Ollama and Pinecone.

How It Works :-

Upload Resume (PDF or TXT)

You’ll be asked to optionally provide a job title.

Skill Extraction

The resume is parsed and skills are extracted using LLM (Gemini).

RAG-Based Question Generation

Each question is generated based on real content using your resume and a Pinecone vector database.

AI Interview Chat Starts

First, the system acts like an HR interviewer and greets you.

Then, it starts asking contextual technical questions one by one.

Interview Ends After 6 Questions

At the end, you'll receive a thank-you message and your interview is complete.


How To Run It Locally (VSCode Recommended)

1. Download the Code
Click the green Code button on GitHub and download as ZIP.

2. Create a Virtual Environment
In VS Code terminal:
python -m venv venv
Activate the environment:

 On Windows:
 venv\Scripts\activate


3.  Install Required Libraries
pip install -r requirements.txt

Make sure requirements.txt contains all the necessary modules.

Set Your API Keys (Already Done in app.py)
No extra config is required the app.py is already pre-configured to use:

Gemini models for embedding and chat model.

Pinecone index named reg with dimension 768 and namespace interview

Run the Application

python app.py
Then go to: http://127.0.0.1:5000/
live link you can visit :- https://ai-live-7hm6.onrender.com/

Please Note
The response speed may be slow (1–2 minutes) after resume upload this is expected because the system uses a local LLM server and vector embedding process.

Be patient during initial parsing  once complete, the chat interface will load automatically.

You’ll be asked 1 HR question and 5 technical questions, then the interview ends.

Output

Upload confirmation

Interactive AI chat window

Live Q&A interview based on your resume

Final message confirming interview completion

