import argparse
import json
import csv
import time
import os
from tqdm import tqdm

# Optional imports for API libraries
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


def get_api_responses(model_name, api_key, base_url, json_path, custom_string, output_csv, method):
    """
    Processes JSON data, gets responses from an API-based model, and saves them to a CSV file.

    Args:
    model_name (str): The name of the API model (e.g., 'gpt-4o', 'deepseek-chat').
    api_key (str): The API key for the service.
    base_url (str, optional): The base URL for the API endpoint.
    json_path (str): Path to the input JSON file.
    custom_string (str): A custom string to prepend to the prompt.
    output_csv (str): Path for the output CSV file.
    method (str): The method to determine which prompt format to use ('nja' or 'None').
    """
    # 1. Configure the correct API client based on the model name and base_url
    client = None
    model_type = None

    # List of model name prefixes that are compatible with the OpenAI client
    openai_compatible_prefixes = ['gpt', 'deepseek']

    if any(prefix in model_name.lower() for prefix in openai_compatible_prefixes):
        if not OpenAI:
            raise ImportError("The 'openai' library is required. Please install it with 'pip install openai'.")
        print(f"Configuring client for OpenAI-compatible model: {model_name}")
        print(f"Using API Base URL: {base_url}")
        client = OpenAI(api_key=api_key, base_url=base_url)
        model_type = 'openai'

    elif 'gemini' in model_name.lower():
        if not genai:
            raise ImportError(
                "The 'google-generativeai' library is required. Please install it with 'pip install google-generativeai'.")
        print(f"Configuring client for Google Gemini model: {model_name}")
        print(f"Using API Endpoint: {base_url}")
        # Gemini uses client_options to set the endpoint
        client_options = {"api_endpoint": base_url} if base_url else None
        genai.configure(api_key=api_key, client_options=client_options)
        client = genai.GenerativeModel(model_name=model_name)
        model_type = 'gemini'

    else:
        print(f"Error: Unsupported model name '{model_name}'. Cannot determine API type.")
        return

    # 2. Read the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_items = len(data)
    print(f"Processing started, found {total_items} records in {json_path}")

    # 3. Prepare the CSV file for writing
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'input_text', 'response', 'response_length', 'status']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 4. Process each item with a tqdm progress bar
        for idx, item in enumerate(tqdm(data, desc=f"Generating Responses ({model_name})", unit="item")):
            prompt = item.get('nja_format', '') if method == 'nja' else item.get('prompt', '')
            if not prompt:
                continue

            input_text = f"User: {custom_string} {prompt}\nAssistant:"
            response_text = ""
            status = "success"

            try:
                # Retry logic for API calls
                for attempt in range(3):
                    try:
                        if model_type == 'openai':
                            chat_completion = client.chat.completions.create(
                                model=model_name,
                                messages=[{"role": "user", "content": input_text}],
                                timeout=300
                            )
                            response_text = chat_completion.choices[0].message.content

                        elif model_type == 'gemini':
                            response = client.generate_content(
                                input_text,
                                request_options={"timeout": 300}
                            )
                            response_text = response.text

                        break  # Success, exit retry loop
                    except Exception as e:
                        if attempt < 2:
                            wait_time = (attempt + 1) * 5
                            print(
                                f"\nAttempt {attempt + 1}/3 failed for item {idx + 1}. Retrying in {wait_time}s... Error: {e}")
                            time.sleep(wait_time)
                        else:
                            raise

            except Exception as e:
                status = f"error: {type(e).__name__}"
                response_text = f"ERROR: {str(e)}"
                print(f"\nFailed to process item {idx + 1} after all retries: {e}")

            writer.writerow({
                'id': idx + 1,
                'input_text': input_text,
                'response': response_text,
                'response_length': len(response_text),
                'status': status
            })

            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get responses from API-based models and save to a CSV file.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the API model (e.g., 'gpt-4o', 'deepseek-chat').")
    parser.add_argument("--api_key", type=str, required=True, help="API key for the service.")
    # Added base_url argument, making it optional
    parser.add_argument("--base_url", type=str, default=None, help="The base URL for the API endpoint.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--custom_string", type=str, default="", help="Custom string to prepend to the input text.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path for the output CSV file.")
    parser.add_argument("--method", type=str, required=True,
                        help="The processing method to use (e.g., 'nja' or 'None').")

    args = parser.parse_args()

    # Pass the new base_url argument to the main function
    get_api_responses(
        args.model_name,
        args.api_key,
        args.base_url,
        args.json_path,
        args.custom_string,
        args.output_csv,
        args.method
    )