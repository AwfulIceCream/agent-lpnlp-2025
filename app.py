from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import gradio as gr

from agent import AgentError, ExaminerAgent
from tools import get_students_list, reset_session

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Application settings
APP_TITLE = "AI Examiner"
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "7860"))

# Custom CSS for a distinctive, modern look
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --primary-color: #6366f1;
    --secondary-color: #8b5cf6;
    --accent-color: #22d3ee;
    --bg-dark: #0f0f23;
    --bg-card: #1a1a2e;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --border-color: #334155;
    --success-color: #10b981;
    --warning-color: #f59e0b;
}

.gradio-container {
    font-family: 'Outfit', sans-serif !important;
    background: linear-gradient(135deg, var(--bg-dark) 0%, #16213e 50%, #0f0f23 100%) !important;
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 2rem 1rem;
    background: linear-gradient(180deg, rgba(99, 102, 241, 0.1) 0%, transparent 100%);
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 1.5rem;
}

.main-header h1 {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.5rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem !important;
}

.main-header p {
    color: var(--text-secondary) !important;
    font-size: 1.1rem !important;
    max-width: 600px;
    margin: 0 auto !important;
}

.config-panel {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
    margin-bottom: 1rem !important;
}

.config-panel label {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
}

.chat-container {
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

.message {
    font-family: 'Outfit', sans-serif !important;
}

.message.user {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
}

.message.bot {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
}

footer {
    display: none !important;
}

.info-box {
    background: rgba(99, 102, 241, 0.1) !important;
    border: 1px solid var(--primary-color) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
    margin-top: 1rem !important;
}

.info-box p {
    color: var(--text-secondary) !important;
    margin: 0 !important;
    font-size: 0.9rem !important;
}

button.primary {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
    border: none !important;
    font-weight: 600 !important;
}

button.primary:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px);
}

button.secondary {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
}
"""

# Easter egg phrase
EASTER_EGG_PHRASE = "бахни сальтуху"

# Type aliases
ChatHistory = list[dict[str, str]]


class AppState:
    """Application state management."""
    
    def __init__(self) -> None:
        self.agent: ExaminerAgent | None = None
    
    def reset(self) -> None:
        """Reset application state."""
        self.agent = None
        reset_session()


# Global application state
app_state = AppState()


def _create_message(role: str, content: str) -> dict[str, str]:
    """Create a chat message dict."""
    return {"role": role, "content": content}


def _validate_api_key(api_key: str) -> tuple[bool, str]:
    """
    Validate API key format.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not api_key:
        return False, "API key is required"
    
    api_key = api_key.strip()
    
    if len(api_key) < 10:
        return False, "API key appears to be too short"
    
    return True, ""


def initialize_agent(provider: str, api_key: str) -> tuple[ChatHistory, str]:
    """
    Initialize the agent and return the greeting.
    
    Args:
        provider: LLM provider name
        api_key: API key for the provider
        
    Returns:
        Tuple of (chat_history, status_message)
    """
    # Validate API key
    is_valid, error = _validate_api_key(api_key)
    if not is_valid:
        logger.warning(f"Invalid API key: {error}")
        return [], f"⚠️ {error}"
    
    try:
        logger.info(f"Initializing agent with provider: {provider}")
        reset_session()
        app_state.agent = ExaminerAgent(provider.lower(), api_key.strip())
        greeting = app_state.agent.get_initial_greeting()
        
        logger.info("Agent initialized successfully")
        return [_create_message("assistant", greeting)], "✅ Agent initialized successfully!"
    
    except AgentError as e:
        logger.error(f"Agent initialization failed: {e}")
        return [], f"❌ Initialization error: {e}"
    except Exception as e:
        logger.exception("Unexpected error during initialization")
        return [], f"❌ Unexpected error: {e}"


def _handle_easter_egg() -> str:
    """Handle the easter egg command."""
    logger.info("Easter egg activated!")
    
    try:
        students = get_students_list()
        if not students:
            return "🤸 *сальтуха* 🤸\n\nБаза студентів порожня."
        
        students_list = "\n".join([
            f"• {s.get('name', 'N/A')} ({s.get('email', 'N/A')})" 
            for s in students
        ])
        return f"🤸 *сальтуха* 🤸\n\n**Зареєстровані студенти:**\n{students_list}"
    except Exception as e:
        logger.error(f"Error loading students: {e}")
        return "🤸 *сальтуха* 🤸\n\nНе вдалося завантажити базу студентів."


def chat(message: str, history: ChatHistory) -> tuple[str, ChatHistory]:
    """
    Handle chat messages.
    
    Args:
        message: User's message
        history: Current chat history
        
    Returns:
        Tuple of (empty_string, updated_history)
    """
    # Handle empty message
    if not message or not message.strip():
        return "", history
    
    # Ensure history is a list
    if history is None:
        history = []
    
    message = message.strip()
    
    # Easter egg check
    if message == EASTER_EGG_PHRASE:
        response = _handle_easter_egg()
        return "", history + [
            _create_message("user", message),
            _create_message("assistant", response)
        ]
    
    # Check if agent is initialized
    if app_state.agent is None:
        logger.warning("Chat attempted without initialized agent")
        return "", history + [
            _create_message("user", message),
            _create_message("assistant", "⚠️ Please initialize the agent first by clicking 'Start Exam'.")
        ]
    
    # Process message through agent
    try:
        logger.debug(f"Processing message: {message[:50]}...")
        response = app_state.agent.chat(message)
        
        return "", history + [
            _create_message("user", message),
            _create_message("assistant", response)
        ]
    except Exception as e:
        logger.exception("Error during chat processing")
        return "", history + [
            _create_message("user", message),
            _create_message("assistant", f"❌ Error: {e}")
        ]


def clear_chat() -> tuple[ChatHistory, str]:
    """
    Clear the chat and reset the session.
    
    Returns:
        Tuple of (empty_history, status_message)
    """
    logger.info("Chat cleared")
    app_state.reset()
    return [], "Chat cleared. Click 'Start Exam' to begin a new exam."


def create_demo() -> gr.Blocks:
    """Create the Gradio interface."""
    
    with gr.Blocks(title=APP_TITLE) as demo:
        # Header
        gr.HTML("""
            <div class="main-header">
                <h1>🎓 AI Examiner</h1>
                <p>An intelligent NLP course examination system powered by LLMs. 
                   Test your knowledge in Natural Language Processing through an interactive dialogue.</p>
            </div>
        """)
        
        with gr.Row():
            # Left panel - Configuration
            with gr.Column(scale=1):
                gr.HTML("<h3 style='color: #e2e8f0; margin-bottom: 1rem;'>⚙️ Configuration</h3>")
                
                provider = gr.Dropdown(
                    choices=["Groq", "Gemini"],
                    value="Groq",
                    label="LLM Provider",
                    info="Select your preferred LLM provider"
                )
                
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder="Enter your API key...",
                    info="Your API key is not stored and only used for this session"
                )
                
                with gr.Row():
                    start_btn = gr.Button("🚀 Start Exam", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                
                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Enter your API key and click 'Start Exam' to begin."
                )
                
                gr.HTML("""
                    <div class="info-box">
                        <p><strong>📋 How it works:</strong></p>
                        <p>1. Enter your Groq or Gemini API key</p>
                        <p>2. Click 'Start Exam' to initialize</p>
                        <p>3. Provide your name and email when asked</p>
                        <p>4. Answer questions about NLP topics</p>
                        <p>5. Receive your score and feedback</p>
                    </div>
                """)
                
                gr.HTML("""
                    <div class="info-box" style="margin-top: 0.5rem;">
                        <p><strong>🔑 Get API Keys:</strong></p>
                        <p>• Groq: <a href="https://console.groq.com" target="_blank" style="color: #22d3ee;">console.groq.com</a></p>
                        <p>• Gemini: <a href="https://aistudio.google.com" target="_blank" style="color: #22d3ee;">aistudio.google.com</a></p>
                    </div>
                """)
            
            # Right panel - Chat
            with gr.Column(scale=2):
                gr.HTML("<h3 style='color: #e2e8f0; margin-bottom: 1rem;'>💬 Exam Chat</h3>")
                
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Response",
                        placeholder="Type your answer here...",
                        scale=4,
                        show_label=False
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
        
        # Event handlers
        start_btn.click(
            fn=initialize_agent,
            inputs=[provider, api_key],
            outputs=[chatbot, status]
        )
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, status]
        )
        
        msg.submit(
            fn=chat,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        send_btn.click(
            fn=chat,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
    
    return demo


# Create the demo instance
demo = create_demo()

if __name__ == "__main__":
    logger.info(f"Starting {APP_TITLE} on {SERVER_HOST}:{SERVER_PORT}")
    
    demo.launch(
        server_name=SERVER_HOST,
        server_port=SERVER_PORT,
        share=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Base()
    )
