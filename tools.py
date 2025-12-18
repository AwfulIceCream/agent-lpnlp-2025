from __future__ import annotations

import json
import logging
import random
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from config import NUM_EXAM_TOPICS

# Configure logging
logger = logging.getLogger(__name__)

# Path to data directory
DATA_DIR = Path(__file__).parent / "data"

# File lock for thread-safe JSON operations
_file_lock = threading.Lock()


class StudentRecord(TypedDict):
    """Type definition for a student record."""
    email: str
    name: str


class ExamRecord(TypedDict):
    """Type definition for an exam record."""
    email: str
    name: str
    score: float
    feedback: str
    topics: list[str]
    start_time: str | None
    end_time: str
    history: list[dict]


class ExamSession:
    """Tracks the current exam session state."""
    
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()
    
    def reset(self) -> None:
        """Reset session to initial state."""
        with self._lock:
            self.email: str | None = None
            self.name: str | None = None
            self.topics: list[str] = []
            self.current_topic_index: int = 0
            self.start_time: datetime | None = None
            self.history: list[dict] = []
    
    @property
    def is_active(self) -> bool:
        """Check if an exam session is active."""
        return bool(self.topics)


# Global session instance
_session = ExamSession()


def get_session() -> ExamSession:
    """Get the current exam session."""
    return _session


def reset_session() -> None:
    """Reset the exam session."""
    _session.reset()
    logger.info("Exam session reset")


def _load_json(filename: str) -> dict[str, Any]:
    """
    Load a JSON file from the data directory.
    
    Args:
        filename: Name of the JSON file
        
    Returns:
        Parsed JSON content or empty dict if file doesn't exist
    """
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        logger.warning(f"File not found: {filepath}")
        return {}
    
    try:
        with _file_lock:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        return {}
    except IOError as e:
        logger.error(f"Error reading {filepath}: {e}")
        return {}


def _save_json(filename: str, data: dict[str, Any]) -> bool:
    """
    Save data to a JSON file in the data directory.
    
    Args:
        filename: Name of the JSON file
        data: Data to save
        
    Returns:
        True if successful, False otherwise
    """
    filepath = DATA_DIR / filename
    
    try:
        with _file_lock:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            # Write to temp file first, then rename for atomicity
            temp_path = filepath.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            temp_path.replace(filepath)
        logger.debug(f"Saved data to {filepath}")
        return True
    except IOError as e:
        logger.error(f"Error writing {filepath}: {e}")
        return False


def _validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid email format
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def _sanitize_string(value: str, max_length: int = 500) -> str:
    """
    Sanitize and truncate a string input.
    
    Args:
        value: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    return value.strip()[:max_length]


def start_exam(email: str, name: str) -> dict[str, Any]:
    """
    Start the exam for a student.
    
    Args:
        email: The student's email address
        name: The student's full name
        
    Returns:
        A dict with 'topics' (list of topic names) on success, or 'error' message
    """
    # Input validation
    email = _sanitize_string(email.lower(), 254)  # Max email length per RFC
    name = _sanitize_string(name, 200)
    
    if not email or not name:
        logger.warning("start_exam called with empty email or name")
        return {"error": "Email and name are required."}
    
    if not _validate_email(email):
        logger.warning(f"Invalid email format: {email}")
        return {"error": "Please provide a valid email address."}
    
    if len(name) < 2:
        return {"error": "Please provide your full name."}
    
    # Load students database
    students_data = _load_json("students.json")
    if "students" not in students_data:
        students_data["students"] = []
    students: list[dict] = students_data["students"]
    
    # Check if student is registered (case-insensitive email match)
    student_found = False
    for student in students:
        if student.get("email", "").lower() == email:
            student_found = True
            # Update name if different
            if student.get("name") != name:
                student["name"] = name
                _save_json("students.json", students_data)
                logger.info(f"Updated student name: {email}")
            break
    
    # Register new student if not found
    if not student_found:
        new_student: StudentRecord = {"email": email, "name": name}
        students_data["students"].append(new_student)
        if not _save_json("students.json", students_data):
            return {"error": "Failed to register student. Please try again."}
        logger.info(f"Registered new student: {email}")
    
    # Load topics
    topics_data = _load_json("topics.json")
    all_topics = topics_data.get("topics", [])
    
    min_topics, max_topics = NUM_EXAM_TOPICS
    if len(all_topics) < min_topics:
        logger.error("Not enough topics in database")
        return {"error": "Not enough topics available for the exam."}
    
    # Randomly select topics
    num_topics = random.randint(min_topics, min(max_topics, len(all_topics)))
    selected_topics = random.sample(all_topics, num_topics)
    topic_names = [t["name"] for t in selected_topics]
    
    # Initialize session
    session = get_session()
    session.email = email
    session.name = name
    session.topics = topic_names
    session.current_topic_index = 0
    session.start_time = datetime.now()
    session.history = []
    
    logger.info(f"Exam started for {email} with topics: {topic_names}")
    
    registration_note = " (newly registered)" if not student_found else ""
    return {
        "success": True,
        "topics": topic_names,
        "message": f"Exam started for {name}{registration_note}. Topics to cover: {', '.join(topic_names)}"
    }


def get_next_topic() -> dict[str, Any]:
    """
    Get the next topic to examine.
    
    Returns:
        A dict with 'topic' name, or 'finished' flag if all topics are done
    """
    session = get_session()
    
    if not session.is_active:
        logger.warning("get_next_topic called without active session")
        return {"error": "No exam in progress. Please start an exam first."}
    
    session.current_topic_index += 1
    
    if session.current_topic_index >= len(session.topics):
        logger.info(f"All topics covered for {session.email}")
        return {
            "finished": True,
            "message": "All topics have been covered. Please provide the final evaluation and end the exam."
        }
    
    next_topic = session.topics[session.current_topic_index]
    logger.debug(f"Moving to topic {session.current_topic_index + 1}: {next_topic}")
    
    return {
        "topic": next_topic,
        "topic_number": session.current_topic_index + 1,
        "total_topics": len(session.topics),
        "message": f"Moving to topic {session.current_topic_index + 1} of {len(session.topics)}: {next_topic}"
    }


def end_exam(email: str, score: float, feedback: str) -> dict[str, Any]:
    """
    End the exam and record the results.
    
    Args:
        email: The student's email address
        score: The final score (0-10)
        feedback: Summary feedback for the student
        
    Returns:
        A dict confirming the exam has been recorded
    """
    session = get_session()
    
    # Input validation
    email = _sanitize_string(email.lower(), 254)
    feedback = _sanitize_string(feedback, 5000)
    
    try:
        score = float(score)
    except (TypeError, ValueError):
        logger.warning(f"Invalid score type: {type(score)}")
        return {"error": "Score must be a number."}
    
    if not 0 <= score <= 10:
        logger.warning(f"Score out of range: {score}")
        return {"error": "Score must be between 0 and 10."}
    
    if not feedback:
        return {"error": "Feedback is required."}
    
    # Create exam record
    exam_record: ExamRecord = {
        "email": email,
        "name": session.name or "Unknown",
        "score": round(score, 1),
        "feedback": feedback,
        "topics": session.topics,
        "start_time": session.start_time.isoformat() if session.start_time else None,
        "end_time": datetime.now().isoformat(),
        "history": session.history
    }
    
    # Load existing results and append
    results_data = _load_json("exam_results.json")
    if "exams" not in results_data:
        results_data["exams"] = []
    
    results_data["exams"].append(exam_record)
    
    if not _save_json("exam_results.json", results_data):
        logger.error(f"Failed to save exam results for {email}")
        return {"error": "Failed to save exam results. Please try again."}
    
    logger.info(f"Exam completed for {email} with score {score}/10")
    
    # Reset session
    reset_session()
    
    return {
        "success": True,
        "message": f"Exam results recorded successfully. Score: {score}/10"
    }


def add_to_history(role: str, content: str) -> None:
    """
    Add a message to the conversation history.
    
    Args:
        role: Message role (user, assistant, system)
        content: Message content
    """
    session = get_session()
    session.history.append({
        "role": role,
        "content": content,
        "datetime": datetime.now().isoformat()
    })


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a tool by name with the given arguments.
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Dictionary of arguments to pass to the tool
        
    Returns:
        The result of the tool execution
    """
    tools = {
        "start_exam": start_exam,
        "get_next_topic": get_next_topic,
        "end_exam": end_exam,
    }
    
    if tool_name not in tools:
        logger.warning(f"Unknown tool called: {tool_name}")
        return {"error": f"Unknown tool: {tool_name}"}
    
    logger.debug(f"Executing tool: {tool_name} with args: {arguments}")
    
    try:
        result = tools[tool_name](**arguments)
        logger.debug(f"Tool {tool_name} result: {result}")
        return result
    except TypeError as e:
        logger.error(f"Invalid arguments for {tool_name}: {e}")
        return {"error": f"Invalid arguments for {tool_name}: {e}"}
    except Exception as e:
        logger.exception(f"Error executing {tool_name}")
        return {"error": f"Error executing {tool_name}: {e}"}


def get_students_list() -> list[StudentRecord]:
    """
    Get list of all registered students.
    
    Returns:
        List of student records
    """
    students_data = _load_json("students.json")
    return students_data.get("students", [])
