import os
import subprocess
import sys

def run_streamlit_app():
    """Run the Streamlit app from the webapp directory."""
    webapp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webapp", "app.py")
    
    if not os.path.exists(webapp_path):
        print(f"Error: Webapp file not found at {webapp_path}")
        return 1
    
    print(f"Starting KnowledgeNexus web app...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", webapp_path, "--server.port=8501"])
    return 0

if __name__ == "__main__":
    sys.exit(run_streamlit_app())