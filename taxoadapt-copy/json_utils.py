import json
import re

def clean_json_string(text):
    """
    Clean and extract JSON from LLM output that may contain markdown code blocks
    or extra text around the JSON.
    """
    # Remove markdown code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # Try to find JSON object/array in the text
    # Look for content between { } or [ ]
    json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    return text.strip()


def safe_parse_json(text, default=None, context=""):
    """
    Safely parse JSON with better error handling and reporting.
    
    Args:
        text: The text to parse as JSON
        default: Default value to return on failure (default: {})
        context: Context description for error messages
    
    Returns:
        Parsed JSON or default value
    """
    if default is None:
        default = {}
    
    try:
        # First try to clean the text
        cleaned = clean_json_string(text)
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"\n{'='*80}")
        print(f"JSON PARSING ERROR {f'({context})' if context else ''}")
        print(f"{'='*80}")
        print(f"Error: {e}")
        print(f"\nRaw text (first 500 chars):")
        print(text[:500])
        print(f"\nCleaned text (first 500 chars):")
        print(cleaned[:500] if 'cleaned' in locals() else "N/A")
        print(f"{'='*80}\n")
        return default
    except Exception as e:
        print(f"\n⚠ Unexpected error parsing JSON {f'({context})' if context else ''}: {e}")
        return default


def retry_on_failure(func, max_retries=3, delay=2):
    """
    Retry a function call on failure with exponential backoff.
    
    Args:
        func: Function to call (should be a lambda or callable)
        max_retries: Maximum number of retry attempts
        delay: Initial delay in seconds (doubles each retry)
    
    Returns:
        Function result or None if all retries failed
    """
    import time
    
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)
                print(f"⚠ Attempt {attempt + 1} failed: {e}")
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"✗ All {max_retries} attempts failed")
                raise
    
    return None
