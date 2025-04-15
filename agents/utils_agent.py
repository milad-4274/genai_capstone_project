import json
import re
import ast 



def extract_json_from_response(text: str) -> dict:
    """
    Parses a variety of LLM outputs into a valid Python dictionary.
    
    Args:
        text (str): LLM output string that should represent a JSON/dict object.
        
    Returns:
        dict: Parsed Python dictionary.
    
    Raises:
        ValueError: If the input cannot be parsed into a dictionary.
    """
    # Step 1: Trim and clean up common LLM issues
    text = text.strip()

    # Step 2: Remove code block markers if present
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()

    # Step 3: Try JSON decoding first
    try:
        return json.loads(text.replace("'", '"'))
    except json.JSONDecodeError:
        msg = "JSON Decode Error"

    # Step 4: Try Python literal_eval
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except (SyntaxError, ValueError):
        msg = "Syntax Error, ast error"

    # Step 5: Try extracting dictionary-like structure with regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return ast.literal_eval(match.group())
        except Exception:
            msg = "Regex Error"

    raise ValueError("Unable to parse LLM output into JSON due " + msg)