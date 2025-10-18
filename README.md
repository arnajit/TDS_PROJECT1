# TDS_PROJECT1

# 🚀 LLM Code Deployment Project

This project implements an **automated build–deploy–revise pipeline** for generating, deploying, and evaluating web applications using Large Language Models (LLMs).  
It hosts an **API endpoint** that accepts structured task requests, uses an LLM to generate code, deploys it to **GitHub Pages**, and notifies an external evaluation service.

---

## 📌 Features

- ✅ Secret verification for incoming requests  
- ⚡ Automated app generation using LLMs  
- 🌐 GitHub Pages deployment with public repositories  
- 📝 Auto-generated MIT LICENSE and professional README  
- 📬 Webhook integration to notify evaluation endpoints  
- 🔄 Round 2 revision support to update apps after instructor feedback

---

## 🧠 Project Workflow

### 1️⃣ Build Phase
- Receives a task request (JSON) via POST  
- Verifies the student secret  
- Parses brief and attachments  
- Generates a minimal application using LLM  
- Creates a public GitHub repository with a unique name  
- Pushes the code with:
  - `index.html` and required assets  
  - MIT LICENSE at root  
  - Professional README  
- Enables GitHub Pages  
- Posts repo metadata to the evaluation URL within 10 minutes

### 2️⃣ Revise Phase
- Receives a **round 2** request to add/modify features  
- Updates the repo accordingly  
- Redeploys to GitHub Pages  
- Notifies the evaluation API again with updated commit and repo details

### 3️⃣ Evaluate Phase (Instructor)
- Instructors run automated checks:
  - Static checks (LICENSE, structure, README etc)
  - LLM-based code and documentation quality analysis
  - Dynamic checks using Playwright

---

## 🧰 Tech Stack

- **Backend**: Python 3.11, FastAPI  
- **LLM Integration**: OpenAI API, AI Pipe  
- **Deployment**: GitHub REST API 
- **Containerization**: Docker  
- **Testing & Evaluation**: LLM-based analysis

---

## 🐳 Running on hugging space at https://arnajit03-tdsp1.hf.space/ (with Docker)
