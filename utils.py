import json
import re
import ast 

# def extract_json_from_response(text):
#     # Extract JSON block from markdown-fenced code
#     match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
#     if match:
#         return json.loads(match.group(1))
#     # Fallback: maybe it's just a plain JSON object without code block
#     elif text.strip().startswith("{") and text.strip().endswith("}"):
#         return json.loads(text.strip())
#     else:
#         raise ValueError("No valid JSON object found in LLM response")


# def extract_json_from_response(llm_response: str) -> dict:
#     """
#     Extracts and parses JSON from an LLM response, handling potential errors.

#     Args:
#         llm_response: The string response from the LLM.

#     Returns:
#         A dictionary representing the parsed JSON, or None if parsing fails.
#     """
#     # 1. Extract JSON using regex
#     json_match = re.search(r"\{[\s\S]*\}", llm_response)
#     if not json_match:
#         print("No JSON found in LLM response.")
#         return None
#     json_str = json_match.group(0)

#     # 2. Attempt to parse the JSON
#     data = json.loads(json_str)
#     return data


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
        pass

    # Step 4: Try Python literal_eval
    try:
        result = ast.literal_eval(text)
        if isinstance(result, dict):
            return result
    except (SyntaxError, ValueError):
        pass

    # Step 5: Try extracting dictionary-like structure with regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return ast.literal_eval(match.group())
        except Exception:
            pass

    raise ValueError("Unable to parse LLM output into JSON")