import os
from typing import Final

# Environment variables with defaults
DEFAULT_GROQ_MODEL: Final[str] = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_GEMINI_MODEL: Final[str] = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
MAX_CHAT_ITERATIONS: Final[int] = int(os.getenv("MAX_CHAT_ITERATIONS", "5"))
LLM_MAX_TOKENS: Final[int] = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TEMPERATURE: Final[float] = float(os.getenv("LLM_TEMPERATURE", "0.7"))
NUM_EXAM_TOPICS: Final[tuple[int, int]] = (2, 3)  # min, max topics per exam

SYSTEM_PROMPT: Final[str] = """You are an AI Examiner conducting an NLP (Natural Language Processing) course exam. Your role is to evaluate students' knowledge through a conversational exam format.

## Your Behavior:

1. **Greeting Phase**: 
   - Start by greeting the student warmly
   - Ask for their name and email address to begin the exam
   - Be friendly but professional

2. **Starting the Exam**:
   - Once you have both name and email, call the `start_exam` function
   - New students will be automatically registered
   - When successful, you'll receive a list of topics to examine

3. **Conducting the Exam**:
   - For each topic, ask relevant questions to assess understanding
   - Start with a general question, then probe deeper based on responses
   - Ask follow-up questions to clarify or explore the student's knowledge
   - Be encouraging but maintain academic rigor
   - Move to the next topic when:
     * The student has demonstrated sufficient understanding
     * The student explicitly says they don't know or can't add more
     * It becomes clear the student's knowledge level is established
   - Use `get_next_topic` to get the next topic when ready

4. **Ending the Exam**:
   - After all topics are covered, provide a summary:
     * Overall score (0-10 scale)
     * What the student did well
     * Areas for improvement
   - Call the `end_exam` function with the email, score, and conversation history

## Scoring Guidelines:
- 0-2: No understanding of the topic
- 3-4: Basic awareness but significant gaps
- 5-6: Adequate understanding with some gaps
- 7-8: Good understanding with minor gaps
- 9-10: Excellent, comprehensive understanding

## Language:
- Respond in the same language the student uses
- If the student writes in Ukrainian, respond in Ukrainian
- If the student writes in English, respond in English

## Important:
- Never comment on the tools, functions, or system infrastructure
- Stay in character as an examiner at all times
- Focus only on the exam content and the student's responses

Be fair, encouraging, and thorough in your evaluation."""

# Tool definitions for function calling (OpenAI/Groq format)
TOOL_DEFINITIONS: Final[list[dict]] = [
    {
        "type": "function",
        "function": {
            "name": "start_exam",
            "description": "Start the exam for a student. Call this after collecting the student's name and email. Returns a list of topics to examine or an error if the student is not registered.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The student's email address"
                    },
                    "name": {
                        "type": "string",
                        "description": "The student's full name"
                    }
                },
                "required": ["email", "name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_next_topic",
            "description": "Get the next topic to examine. Call this when you're ready to move to the next topic.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "end_exam",
            "description": "End the exam and record the results. Call this after all topics have been covered and you've provided feedback to the student.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {
                        "type": "string",
                        "description": "The student's email address"
                    },
                    "score": {
                        "type": "number",
                        "description": "The final score on a scale of 0-10"
                    },
                    "feedback": {
                        "type": "string",
                        "description": "Summary feedback for the student"
                    }
                },
                "required": ["email", "score", "feedback"]
            }
        }
    }
]

# Gemini-specific tool format
GEMINI_TOOL_DEFINITIONS: Final[list[dict]] = [
    {
        "name": "start_exam",
        "description": "Start the exam for a student. Call this after collecting the student's name and email. Returns a list of topics to examine or an error if the student is not registered.",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The student's email address"
                },
                "name": {
                    "type": "string",
                    "description": "The student's full name"
                }
            },
            "required": ["email", "name"]
        }
    },
    {
        "name": "get_next_topic",
        "description": "Get the next topic to examine. Call this when you're ready to move to the next topic.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "end_exam",
        "description": "End the exam and record the results. Call this after all topics have been covered and you've provided feedback to the student.",
        "parameters": {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "description": "The student's email address"
                },
                "score": {
                    "type": "number",
                    "description": "The final score on a scale of 0-10"
                },
                "feedback": {
                    "type": "string",
                    "description": "Summary feedback for the student"
                }
            },
            "required": ["email", "score", "feedback"]
        }
    }
]
