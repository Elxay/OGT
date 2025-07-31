import argparse
import csv
import json
import time
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    print("The 'openai' library is required for this script. Please install it with 'pip install openai'.")
    exit(1)


def evaluate_responses(input_csv, prompt_file, output_csv, api_key, base_url):
    """
    Evaluates model responses in a CSV file using GPT-4o and saves the results.

    Args:
    input_csv (str): Path to the input CSV file containing model responses.
    prompt_file (str): Path to the .txt file containing the evaluation prompt template.
    output_csv (str): Path for the output CSV file with evaluation results.
    api_key (str): Your OpenAI API key.
    base_url (str, optional): The base URL for the OpenAI API endpoint.
    """
    # 1. Initialize the OpenAI client
    print(f"Initializing OpenAI client with model 'gpt-4o'...")
    if not base_url:
        base_url = "https://api.openai.com/v1"  # Default to official OpenAI endpoint

    client = OpenAI(api_key=api_key, base_url=base_url)
    print(f"Using API Base URL: {base_url}")

    # 2. Read the evaluation prompt template
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            eval_prompt_template = f.read()
        print(f"Successfully loaded evaluation prompt from: {prompt_file}")
    except FileNotFoundError:
        print(f"Error: Evaluation prompt file not found at '{prompt_file}'")
        return

    # 3. Read the input CSV data
    try:
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            original_data = list(reader)
            original_fieldnames = reader.fieldnames
        print(f"Found {len(original_data)} records to evaluate in {input_csv}.")
    except FileNotFoundError:
        print(f"Error: Input CSV file not found at '{input_csv}'")
        return

    # 4. Prepare the output CSV file
    new_fieldnames = original_fieldnames + ['evaluation_score', 'evaluation_reasoning']
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=new_fieldnames)
        writer.writeheader()

        # 5. Process each row, evaluate, and write to the new file
        for row in tqdm(original_data, desc="Evaluating Responses", unit="row"):
            original_prompt = row.get('input_text', '')
            model_response = row.get('response', '')

            if not model_response or model_response.startswith("ERROR:"):
                # Skip rows with no response or with processing errors
                eval_score = "skipped"
                eval_reasoning = "Original response was empty or an error."
            else:
                # Construct the full prompt for the evaluator model
                full_eval_prompt = eval_prompt_template.format(
                    original_prompt=original_prompt,
                    model_response=model_response
                )

                try:
                    response_text = ""
                    for attempt in range(3):
                        try:
                            chat_completion = client.chat.completions.create(
                                model="gpt-4o",
                                messages=[{"role": "user", "content": full_eval_prompt}],
                                response_format={"type": "json_object"},
                                temperature=0.0,  # Low temperature for consistent evaluation
                                timeout=120
                            )
                            response_text = chat_completion.choices[0].message.content
                            break  # Success
                        except Exception as e:
                            if attempt < 2:
                                time.sleep((attempt + 1) * 3)
                            else:
                                raise

                    # Parse the JSON response from the evaluator
                    eval_result = json.loads(response_text)
                    eval_score = eval_result.get('score', 'parse_error')
                    eval_reasoning = eval_result.get('reasoning', 'parse_error')

                except Exception as e:
                    print(f"\nAn error occurred while evaluating row {row.get('id', 'N/A')}: {e}")
                    eval_score = "error"
                    eval_reasoning = str(e)

            # Add evaluation results to the row and write to the new CSV
            row['evaluation_score'] = eval_score
            row['evaluation_reasoning'] = eval_reasoning
            writer.writerow(row)

            # Rate limiting to be kind to the API
            time.sleep(1)

    print(f"\nEvaluation complete. Results saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model responses in a CSV using GPT-4o.")
    parser.add_argument("--input-csv", required=True, help="Path to the input CSV file with model responses.")
    parser.add_argument("--prompt-file", required=True,
                        help="Path to the text file containing the evaluation prompt template.")
    parser.add_argument("--output-csv", required=True, help="Path for the output CSV file to save evaluation results.")
    parser.add_argument("--api-key", required=True, help="Your OpenAI API key.")
    parser.add_argument("--base-url", default=None, help="Optional: The base URL for the OpenAI API endpoint.")

    args = parser.parse_args()

    evaluate_responses(
        args.input_csv,
        args.prompt_file,
        args.output_csv,
        args.api_key,
        args.base_url
    )