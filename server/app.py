import threading
import uvicorn
from main import app as fastapi_app
import gradio as gr

# Run FastAPI in background thread
def run_fastapi():
    uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)

def main():
    thread = threading.Thread(target=run_fastapi, daemon=True)
    thread.start()

    # Minimal Gradio front-end so HF is satisfied
    with gr.Blocks() as demo:
        gr.Markdown("# Clinical Triage OpenEnv API is running on port 7860")

    demo.launch(server_port=7861)

if __name__ == "__main__":
    main()