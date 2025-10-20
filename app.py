# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi[standard]",
#   "uvicorn",
#   "requests",
#   "python-dotenv",
#   "PyGithub",
#   "pillow"
# ]
# ///

import requests
from dotenv import load_dotenv
import os
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import base64
import time
from typing import Optional
from PIL import Image
import io
import re
from datetime import datetime

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
SECRET_KEY = os.getenv("SECRET")
AI_PIPE_TOKEN = os.getenv("AI_PIPE_TOKEN")
AI_PIPE_URL = os.getenv("OPENAI_BASE_URL", "https://aipipe.org/openai/v1")
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "arnajit")

if not AI_PIPE_TOKEN:
    raise RuntimeError("AI_PIPE_TOKEN not set in .env")
  
app = FastAPI()


def get_existing_file(repo_name: str, filename: str) -> Optional[str]:
    """Fetch existing file content from GitHub repo"""
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    try:
        response = requests.get(
            f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{filename}",
            headers=headers
        )
        
        if response.status_code == 200:
            # Decode the base64 content
            content_b64 = response.json().get("content", "")
            content = base64.b64decode(content_b64).decode('utf-8')
            return content
    except Exception as e:
        print(f"Failed to fetch existing {filename}: {e}")
    
    return None

def process_attachments(attachments: list) -> tuple:
    """Process attachments with improved error handling"""
    attachment_texts = []
    attachment_files = []
    processed_attachments = []  # For vision API
    
    for att in attachments:
        try:
            filename = att.get("name", "unknown")
            data_uri = att.get("url", "")
            
            # Validate data URI format
            if not data_uri.startswith("data:"):
                print(f"Invalid data URI for {filename}")
                continue
            
            # Split on first comma only
            if "," not in data_uri:
                print(f"Malformed data URI for {filename}")
                continue
                
            header, encoded = data_uri.split(",", 1)
            
            # Decode base64 with error handling
            try:
                # Add padding if needed
                missing_padding = len(encoded) % 4
                if missing_padding:
                    encoded += '=' * (4 - missing_padding)
                content_bytes = base64.b64decode(encoded)
            except Exception as e:
                print(f"Failed to decode base64 for {filename}: {e}")
                continue
            
            # Add to files list
            attachment_files.append({
                "name": filename,
                "content": content_bytes
            })
            
            # Process text files for LLM context
            if filename.endswith((".txt", ".csv", ".md", ".json")):
                try:
                    content_text = content_bytes.decode('utf-8', errors='ignore')
                    max_chars = 5000
                    if len(content_text) > max_chars:
                        content_text = content_text[:max_chars] + f"\n... (truncated, total: {len(content_text)} chars)"
                    attachment_texts.append(f"File: {filename}\n{content_text}")
                except Exception as e:
                    print(f"Failed to decode text from {filename}: {e}")
            
            # Keep images for both vision API and embedding
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")):
                processed_attachments.append({
                    "name": filename,
                    "url": data_uri,  # Original data URI for embedding
                    "content": content_bytes  # Raw bytes for compression/analysis
                })
                
        except Exception as e:
            print(f"Failed to process attachment {att.get('name', 'unknown')}: {e}")
    
    return attachment_texts, attachment_files, processed_attachments


def validate_secret(secret: str) -> bool:
    return secret == SECRET_KEY

def create_github_repo(repo_name: str) -> dict:
    """Create a public GitHub repo with MIT license"""
    payload = {
        "name": repo_name,
        "private": False, 
        "auto_init": True,
        "license_template": "mit"
    }  
    
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }

    response = requests.post(
        "https://api.github.com/user/repos", 
        headers=headers, 
        json=payload
    )

    if response.status_code == 422:
        # Repo might already exist
        print(f"Repo {repo_name} may already exist")
        return {"name": repo_name, "html_url": f"https://github.com/{GITHUB_USERNAME}/{repo_name}"}
    elif response.status_code != 201:
        raise Exception(f"Failed to create repo: {response.status_code}, {response.text}")
    
    return response.json()

def enable_github_pages(repo_name: str, max_retries: int = 5):
    """Enable GitHub Pages with retry logic"""
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    payload = {
        "build_type": "legacy",
        "source": {"branch": "main", "path": "/"}
    }
    
    for attempt in range(max_retries):
        response = requests.post(
            f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/pages", 
            headers=headers, 
            json=payload
        )
        
        if response.status_code == 201:
            return True
        elif response.status_code == 409:
            # Pages already enabled
            print(f"GitHub Pages already enabled for {repo_name}")
            return True
        
        # Wait before retry (GitHub needs time after repo creation)
        time.sleep(2 ** attempt)
    
    raise Exception(f"Failed to enable GitHub Pages after {max_retries} attempts")

def get_file_sha(repo_name: str, path: str) -> Optional[str]:
    """Get SHA of a file if it exists"""
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    response = requests.get(
        f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{path}",
        headers=headers
    )
    
    if response.status_code == 200:
        return response.json().get("sha")
    return None

def push_files_to_repo(repo_name: str, files: list[dict]):
    """Push files to GitHub repo"""
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    for file in files:
        file_name = file.get("name")
        file_content = file.get("content")
        
        if not file_name or not file_content:
            continue
        
        # Encode content to base64
        if isinstance(file_content, bytes):
            encoded_content = base64.b64encode(file_content).decode('utf-8')
        else:
            encoded_content = base64.b64encode(file_content.encode('utf-8')).decode('utf-8')

        payload = {
            "message": f"Add {file_name}",
            "content": encoded_content
        }
        
        # Check if file exists and get its SHA for update
        existing_sha = get_file_sha(repo_name, file_name)
        if existing_sha:
            payload["sha"] = existing_sha

        response = requests.put(
            f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{file_name}",
            headers=headers,
            json=payload
        )
        
        if response.status_code not in (200, 201):
            print(f"Failed to push {file_name}: {response.status_code}, {response.text}")




    

def get_latest_commit_sha(repo_name: str, branch: str = "main") -> str:
    """Get SHA of the latest commit"""
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    response = requests.get(
        f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/commits/{branch}",
        headers=headers
    )
    
    if response.status_code != 200:
        raise Exception(f"Failed to get latest commit: {response.status_code}, {response.text}")
    
    return response.json()["sha"]


def get_existing_readme(repo_name: str) -> Optional[str]:
    """Fetch existing README content from GitHub repo"""
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json"
    }
    
    try:
        response = requests.get(
            f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/README.md",
            headers=headers
        )
        
        if response.status_code == 200:
            # Decode the base64 content
            content_b64 = response.json().get("content", "")
            content = base64.b64decode(content_b64).decode('utf-8')
            return content
    except Exception as e:
        print(f"Failed to fetch existing README: {e}")
    
    return None


def compress_image_bytes(img_bytes, size=(256, 256), quality=75):
    """Compress image to a small thumbnail for LLM analysis"""
    img = Image.open(io.BytesIO(img_bytes))
    img = img.convert("RGB")
    img.thumbnail(size)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def create_readme(task_name: str, brief: str, repo_url: str, round_num: int = 1, existing_readme: str = None) -> str:
    """Generate a professional README.md with DYNAMIC round continuation"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    if round_num == 1 or not existing_readme:
        # Round 1: Create fresh README with Development History section
        return f"""# {task_name}

## Summary
{brief}

## Setup
1. Clone the repository:
   ```bash
   git clone {repo_url}.git
   cd {task_name}
   ```

2. Open `index.html` in your browser or visit the GitHub Pages URL.

## Usage
Visit the live application at: https://{GITHUB_USERNAME}.github.io/{task_name}/

## Code Explanation
This project is a single-page web application built with HTML, CSS, and JavaScript. It follows the requirements specified in the brief and implements all requested functionality.

The application uses:
- Modern HTML5 structure
- Responsive CSS styling
- Vanilla JavaScript for interactivity
- External libraries loaded via CDN when needed

## Development History

### Round 1 - Initial Implementation ({timestamp})
**Brief:** {brief}

**Features Implemented:**
- Initial application setup
- Core functionality as per requirements
- GitHub Pages deployment
- Responsive design

## License
This project is licensed under the MIT License - see the LICENSE file for details.
"""
    
    else:
        # Round 2+: DYNAMICALLY append to existing README
        
        # Detect the current highest round number
        existing_rounds = re.findall(r'### Round (\d+)', existing_readme)
        highest_round = max([int(r) for r in existing_rounds]) if existing_rounds else 1
        next_round = highest_round + 1
        
        # Find insertion point (before License section)
        if "## License" in existing_readme:
            parts = existing_readme.split("## License")
            main_content = parts[0].rstrip()
            license_section = "\n\n## License" + parts[1]
        else:
            # No license section, append at end
            main_content = existing_readme.rstrip()
            license_section = "\n\n## License\nThis project is licensed under the MIT License - see the LICENSE file for details.\n"
        
        # Ensure Development History section exists
        if "## Development History" not in main_content:
            main_content += "\n\n## Development History\n"
        
        # Create new round section dynamically
        new_round_section = f"""

### Round {next_round} - Enhancement ({timestamp})
**Brief:** {brief}

**Updates Made:**
- Enhanced functionality based on new requirements
- Additional features implemented
- Code refactored for improved performance
- UI/UX improvements
"""
        
        return main_content + new_round_section + license_section


def write_code_with_llm(brief: str, attachments: list = None, model: str = "gpt-4-turbo", existing_code: str = None) -> tuple:
    """
    Generate web app code with intelligent image handling.
    Determines whether to analyze images or embed them based on brief keywords.
    If existing_code is provided, updates it instead of creating new code.
    Returns (html_files, attachment_files)
    """
    # Process attachments
    if attachments:
        attachment_texts, attachment_files, processed_attachments = process_attachments(attachments)
    else:
        attachment_texts, attachment_files, processed_attachments = [], [], []

    full_brief = brief
    if attachment_texts:
        full_brief += "\n\nAttached text files:\n" + "\n\n".join(attachment_texts)

    # Separate images and other files
    image_attachments = [att for att in processed_attachments if att["name"].lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"))]
    image_files = [att["name"] for att in image_attachments]
    other_files = [f["name"] for f in attachment_files if f["name"] not in image_files]

    # Intelligent detection: should we analyze images or just embed them?
    vision_keywords = [
        "analyze", "describe", "what", "identify", "recognize", "detect",
        "read", "extract", "ocr", "text in", "content of", "caption",
        "tell me about", "what's in", "what is in", "explain the image",
        "image contains", "figure out", "interpret", "examine"
    ]
    
    embedding_keywords = [
        "display", "show", "embed", "include", "add", "place", "put",
        "gallery", "carousel", "slideshow", "layout", "design", "website",
        "webpage", "page with", "use the image", "insert"
    ]
    
    brief_lower = brief.lower()
    has_vision_keywords = any(k in brief_lower for k in vision_keywords)
    has_embedding_keywords = any(k in brief_lower for k in embedding_keywords)
    
    # Decision logic
    use_vision = has_vision_keywords and len(image_files) > 0
    embed_images = (has_embedding_keywords or not has_vision_keywords) and len(image_files) > 0
    
    # Build file information
    if image_files or other_files:
        full_brief += "\n\n⚠️ AVAILABLE FILES ⚠️"
        
        if image_files:
            if use_vision:
                full_brief += f"\n📸 Images to analyze: {', '.join(image_files)}"
                full_brief += "\n   (Vision analysis will be performed)"
            
            if embed_images:
                full_brief += f"\n🖼️ Images available for embedding in the website:"
                for img in image_files:
                    full_brief += f"\n  • {img} → use <img src='{img}' alt='description'>"
                full_brief += "\n\n⚠️ IMPORTANT: Use EXACT filenames when referencing images!"
                full_brief += "\n⚠️ Images will be available in the same directory as index.html"
        
        if other_files:
            full_brief += f"\n📄 Other files available: {', '.join(other_files)}"

    # Build prompt based on whether we're updating or creating
    if existing_code:
        prompt = f"""You are updating an existing web application. Here is the current code:

<existing_code>
{existing_code}
</existing_code>

NEW REQUIREMENTS:
{full_brief}

CRITICAL REQUIREMENTS:
- UPDATE the existing code to incorporate the new requirements
- PRESERVE all existing functionality unless explicitly asked to change it
- ADD new features requested in the brief
- Maintain the same structure and style
- Reference image files by their EXACT filenames (they will be in the same directory)
- NO placeholders or dummy content - everything must be functional
- Images should use relative paths: <img src="filename.png">
- Ensure responsive design and modern UI
- Include proper error handling

Return ONLY the COMPLETE UPDATED HTML code, no markdown formatting."""
    else:
        prompt = f"""Generate a complete, functional single-file web app (HTML with embedded CSS and JavaScript):

{full_brief}

CRITICAL REQUIREMENTS:
- Complete working HTML file with embedded CSS/JS
- Load external libraries from CDN if needed (Bootstrap, jQuery, etc.)
- Reference image files by their EXACT filenames (they will be in the same directory)
- NO placeholders or dummy content - everything must be functional
- Images should use relative paths: <img src="filename.png">
- Ensure responsive design and modern UI
- Include proper error handling

Return ONLY the HTML code, no markdown formatting."""

    messages = [
        {"role": "system", "content": "You are an expert web developer. Generate clean, production-ready HTML with embedded CSS and JavaScript. Always reference files by their exact filenames. When updating code, preserve existing functionality and add new features seamlessly."},
        {"role": "user", "content": prompt}
    ]

    # Add vision analysis if needed (compressed thumbnails for LLM)
    if use_vision and image_attachments:
        for att in image_attachments:
            try:
                # Create small compressed thumbnail for analysis
                thumb_bytes = compress_image_bytes(att["content"], size=(512, 512), quality=85)
                thumb_b64 = base64.b64encode(thumb_bytes).decode()
                thumb_uri = f"data:image/jpeg;base64,{thumb_b64}"
                
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze this image (filename: {att['name']}):"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": thumb_uri}
                        }
                    ]
                })
            except Exception as e:
                print(f"Failed to add image {att['name']} for vision analysis: {e}")

    # Send request to LLM
    try:
        action = "Updating" if existing_code else "Generating"
        print(f"{action} code with LLM (vision: {use_vision}, embed: {embed_images})...")
        resp = requests.post(
            f"{AI_PIPE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {AI_PIPE_TOKEN}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                "max_tokens": 2000,
                "temperature": 0.3
            },
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
        html_code = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Clean markdown if present
        if "```html" in html_code:
            html_code = html_code.split("```html")[1].split("```")[0].strip()
        elif "```" in html_code:
            parts = html_code.split("```")
            if len(parts) >= 2:
                html_code = parts[1].strip()

        if not html_code or len(html_code) < 100:
            raise ValueError("Generated HTML too short or empty")

        print(f"Successfully {action.lower()} HTML ({len(html_code)} chars)")
        return [{"name": "index.html", "content": html_code}], attachment_files

    except Exception as e:
        print(f"LLM request failed: {e}")
        # If updating failed, return existing code
        if existing_code:
            print("Returning existing code due to LLM failure")
            return [{"name": "index.html", "content": existing_code}], attachment_files
        
        # Fallback HTML with image embedding if available
        fallback_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Application</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { padding: 20px; }
        .error { color: #dc3545; }
        .image-gallery img { max-width: 100%; margin: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Application</h1>
        <p class="error">Error generating content. Please check the console.</p>
"""
        if image_files:
            fallback_html += '        <div class="image-gallery">\n'
            for img in image_files:
                fallback_html += f'            <img src="{img}" alt="{img}" class="img-fluid">\n'
            fallback_html += '        </div>\n'
        
        fallback_html += """    </div>
</body>
</html>"""
        return [{"name": "index.html", "content": fallback_html}], attachment_files




def notify_evaluation_api(evaluation_url: str, payload: dict, max_retries: int = 5):
    """Send notification to evaluation API with exponential backoff"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                evaluation_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                print(f"Successfully notified evaluation API: {payload}")
                return True
            
            print(f"Evaluation API returned {response.status_code}: {response.text}")
            
        except Exception as e:
            print(f"Failed to notify evaluation API (attempt {attempt + 1}): {e}")
        
        # Exponential backoff: 1, 2, 4, 8 seconds
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    
    print(f"Failed to notify evaluation API after {max_retries} attempts")
    return False

ROUND_1_EVAL_DELAY = 90  # Default 90 seconds
ROUND_2_EVAL_DELAY = 90 

def process_round_1(data: dict):
    """Process round 1 task"""
    try:
        brief = data.get("brief", "")
        attachments = data.get("attachments", [])
        task_name = f"{data['task']}"
        
        # Generate code with LLM - intelligent image handling
        html_files, attachment_files = write_code_with_llm(brief, attachments=attachments)
        files = html_files + attachment_files 
        
        # Create README (Round 1)
        repo_url = f"https://github.com/{GITHUB_USERNAME}/{task_name}"
        readme_content = create_readme(task_name, brief, repo_url, round_num=1)
        files.append({"name": "README.md", "content": readme_content})
        
        # Create repo
        print(f"Creating repo: {task_name}")
        repo_data = create_github_repo(task_name)
        
        # Wait a bit for repo to be ready
        time.sleep(2)
        
        # Push files
        print(f"Pushing files to {task_name}")
        push_files_to_repo(task_name, files)
        
        # Wait for files to be committed
        time.sleep(3)
        
        # Enable GitHub Pages
        print(f"Enabling GitHub Pages for {task_name}")
        enable_github_pages(task_name)
        
        # Get latest commit SHA
        time.sleep(2)
        commit_sha = get_latest_commit_sha(task_name)
        
        # Prepare evaluation payload
        pages_url = f"https://{GITHUB_USERNAME}.github.io/{task_name}/"
        eval_payload = {
            "email": data.get("email"),
            "task": data.get("task"),
            "round": 1,
            "nonce": data.get("nonce"),
            "repo_url": repo_url,
            "commit_sha": commit_sha,
            "pages_url": pages_url
        }
        
        # WAIT for GitHub Pages to build and deploy
        print(f"Waiting {ROUND_1_EVAL_DELAY} seconds for GitHub Pages to deploy...")
        time.sleep(ROUND_1_EVAL_DELAY)

        # Notify evaluation API
        evaluation_url = data.get("evaluation_url")
        if evaluation_url:
            print(f"Notifying evaluation API: {evaluation_url}")
            notify_evaluation_api(evaluation_url, eval_payload)
        
        print(f"Round 1 completed for {task_name}")
        
    except Exception as e:
        print(f"Error in round 1 processing: {e}")
        raise

    
def process_round_2(data: dict):
    """Process round 2+ task (revisions) - DYNAMICALLY detects round number and UPDATES existing code"""
    try:
        brief = data.get("brief", "")
        attachments = data.get("attachments", [])
        task_name = f"{data['task']}"
        round_num = data.get("round", 2)  # Get actual round number
        
        # FETCH EXISTING index.html
        existing_html = get_existing_file(task_name, "index.html")
        print(f"Fetched existing index.html: {'Yes' if existing_html else 'No'} ({len(existing_html) if existing_html else 0} chars)")
        
        # Generate updated code with LLM - passes existing code for updating
        html_files, attachment_files = write_code_with_llm(
            brief, 
            attachments=attachments, 
            existing_code=existing_html
        )
        files = html_files + attachment_files
        
        # DYNAMICALLY fetch existing README and continue it
        repo_url = f"https://github.com/{GITHUB_USERNAME}/{task_name}"
        existing_readme = get_existing_readme(task_name)
        
        print(f"Fetched existing README: {'Yes' if existing_readme else 'No'}")
        
        # Create updated README with DYNAMIC continuation
        readme_content = create_readme(
            task_name, 
            brief, 
            repo_url, 
            round_num=round_num, 
            existing_readme=existing_readme
        )
        files.append({"name": "README.md", "content": readme_content})
        
        # Push updated files
        print(f"Updating files in {task_name} (Round {round_num})")
        push_files_to_repo(task_name, files)
        
        # Wait for changes to be committed
        time.sleep(3)
        
        # Get latest commit SHA
        commit_sha = get_latest_commit_sha(task_name)
        
        # Prepare evaluation payload
        pages_url = f"https://{GITHUB_USERNAME}.github.io/{task_name}/"
        eval_payload = {
            "email": data.get("email"),
            "task": data.get("task"),
            "round": round_num,
            "nonce": data.get("nonce"),
            "repo_url": repo_url,
            "commit_sha": commit_sha,
            "pages_url": pages_url
        }
        
        # WAIT for GitHub Pages to rebuild and deploy
        print(f"Waiting {ROUND_2_EVAL_DELAY} seconds for GitHub Pages to deploy...")
        time.sleep(ROUND_2_EVAL_DELAY)

        # Notify evaluation API
        evaluation_url = data.get("evaluation_url")
        if evaluation_url:
            print(f"Notifying evaluation API: {evaluation_url}")
            notify_evaluation_api(evaluation_url, eval_payload)
        
        print(f"Round {round_num} completed for {task_name}")
        
    except Exception as e:
        print(f"Error in round {round_num} processing: {e}")
        raise

@app.post("/handle_task")
async def handle_task(data: dict, background_tasks: BackgroundTasks):
    """Main endpoint to handle task requests"""
    # Validate secret
    if not validate_secret(data.get("secret", "")):
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Invalid secret"}
        )
    
    # Validate required fields
    required_fields = ["email", "task", "round", "nonce", "brief"]
    for field in required_fields:
        if field not in data:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": f"Missing required field: {field}"}
            )
    
    # Process based on round
    round_num = data.get("round")
    
    if round_num == 1:
        # Process in background to return 200 immediately
        background_tasks.add_task(process_round_1, data)
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": "Round 1 processing started"}
        )
    elif round_num >= 2:
        # Process in background to return 200 immediately
        background_tasks.add_task(process_round_2, data)
        return JSONResponse(
            status_code=200,
            content={"status": "success", "message": f"Round {round_num} processing started"}
        )
    else:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": "Invalid round number"}
        )

@app.get("/test")
def test():
    """Test endpoint"""
    return {"status": "ok", "message": "Everything good"}

@app.get("/")
def root():
    """Root endpoint"""
    return {"status": "running", "message": "LLM Code Deployment API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
