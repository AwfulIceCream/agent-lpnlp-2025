# 🎓 AI Examiner Agent

An intelligent NLP course examination system powered by LLMs. This agent conducts oral exams through a conversational interface, evaluating students' knowledge in Natural Language Processing topics.

## Features

- **Interactive Exam Format**: Conducts exams through natural dialogue
- **Multiple LLM Support**: Works with Groq (Llama 3.3) or Google Gemini
- **Function Calling**: Uses LLM tool calling for exam flow control
- **Automatic Scoring**: Provides scores (0-10) and detailed feedback
- **Auto-Registration**: New students are automatically registered
- **Bilingual Support**: Responds in Ukrainian or English based on student's language
- **Production Ready**: Includes logging, error handling, input validation

## NLP Topics Covered

- Tokenization and Text Preprocessing
- Word Embeddings (Word2Vec, GloVe, FastText)
- Recurrent Neural Networks (RNN, LSTM, GRU)
- Attention Mechanism
- Transformer Architecture
- BERT and Masked Language Modeling
- GPT and Autoregressive Models
- Named Entity Recognition (NER)
- Text Classification
- Machine Translation
- Question Answering
- Language Models and Perplexity

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Locally

```bash
python app.py
```

The app will be available at `http://localhost:7860`

### 3. Get API Keys

- **Groq**: Get your free API key at [console.groq.com](https://console.groq.com)
- **Gemini**: Get your API key at [aistudio.google.com](https://aistudio.google.com)

## How to Use

1. **Select Provider**: Choose between Groq or Gemini
2. **Enter API Key**: Paste your API key (it's only used for the session, not stored)
3. **Start Exam**: Click the "Start Exam" button
4. **Provide Credentials**: Enter your name and email when asked (new students are automatically registered)
5. **Answer Questions**: Respond to questions about NLP topics
6. **Get Results**: Receive your score and feedback at the end

## Configuration

The application supports the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `7860` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model to use |
| `GEMINI_MODEL` | `gemini-2.0-flash-exp` | Gemini model to use |
| `LLM_MAX_TOKENS` | `2048` | Maximum tokens in LLM response |
| `LLM_TEMPERATURE` | `0.7` | LLM temperature setting |
| `MAX_CHAT_ITERATIONS` | `5` | Max tool call iterations per message |

## Project Structure

```
agent-lpnlp-2025/
├── app.py              # Gradio web interface
├── agent.py            # LLM agent with function calling
├── tools.py            # Exam tool functions
├── llm_client.py       # Groq/Gemini client abstraction
├── config.py           # Configuration and prompts
├── data/
│   ├── students.json   # Registered students
│   ├── topics.json     # NLP exam topics
│   └── exam_results.json  # Exam history
├── requirements.txt
├── .gitignore
└── README.md
```

## Deployment to HuggingFace Spaces

1. Create a new Space on [HuggingFace](https://huggingface.co/spaces)
2. Select "Gradio" as the SDK
3. Upload all project files
4. The app will automatically deploy

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Gradio Interface                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ API Key     │  │ Provider    │  │ Chat Interface      │  │
│  │ Input       │  │ Selector    │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Examiner Agent                            │
│  ┌─────────────────┐  ┌────────────────────────────────┐    │
│  │ Conversation    │  │ LLM Client (Groq/Gemini)       │    │
│  │ Manager         │  │ with Function Calling          │    │
│  └─────────────────┘  └────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Exam Tools                              │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │start_exam  │  │get_next_topic│  │end_exam            │   │
│  └────────────┘  └──────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    File Storage                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐     │
│  │students    │  │topics      │  │exam_results        │     │
│  │.json       │  │.json       │  │.json               │     │
│  └────────────┘  └────────────┘  └────────────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Error Handling

The application includes comprehensive error handling:

- **Input Validation**: Email format, name length, score range
- **File Operations**: Atomic writes, thread-safe access
- **LLM Errors**: Connection issues, invalid responses
- **Graceful Degradation**: Fallback responses when LLM fails

## Logging

Logs are written to stdout with the following format:
```
2025-01-15 10:30:45 - agent - INFO - Agent initialized with provider: groq
```

Set `LOG_LEVEL=DEBUG` for detailed debugging information.

## License

This project was created for the NLP course at Lviv Polytechnic National University.