import json
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse
from tqdm import tqdm


def save_responses_to_csv(json_path, custom_string, model_path, output_csv, method=None):
    """
    Processes JSON data, gets model responses, and saves them to a CSV file.

    Args:
    json_path (str): Path to the input JSON file.
    custom_string (str): A custom string to prepend to the prompt.
    model_path (str): Path to the local Hugging Face model.
    output_csv (str): Path for the output CSV file.
    method (str, optional): The method to determine which prompt format to use. Defaults to None.
    """
    # 1. Load the model and tokenizer
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True
    ).eval()
    print("Model loaded successfully.")

    # 2. Read the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_items = len(data['data'])
    print(f"Processing started, found {total_items} records in {json_path}")

    # 3. Prepare the CSV file for writing
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'input_text', 'response', 'response_length']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 4. Process each item with a tqdm progress bar
        for idx, item in enumerate(tqdm(data, desc="Generating Responses", unit="item")):
            prompt = ""
            if method == 'nja':
                # Extract the 'nja_format' field for the 'nja' method
                prompt = item.get('nja_format', '')
                if not prompt:
                    # If nja_format is empty, skip this item
                    continue
            else:
                # For any other method (including 'None'), use the 'prompt' field
                prompt = item.get('prompt', '')

            # Construct the final input text for the model
            # This format is suitable for many instruction-tuned models.
            input_text = f"User: {custom_string} {prompt}\nAssistant:"

            try:
                # Generate the model response
                inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4096,  # Increased limit for longer responses
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=tokenizer.eos_token_id
                )
                full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Extract only the generated part of the response (stripping the input)
                generated_response = full_response[len(input_text):].strip()
                response_length = len(generated_response)
            except Exception as e:
                generated_response = f"ERROR: {str(e)}"
                response_length = 0
                print(f"\nAn error occurred while processing item {idx + 1}: {e}")

            # Write the results to the CSV row
            writer.writerow({
                'id': idx + 1,
                'input_text': input_text,
                'response': generated_response,
                'response_length': response_length
            })


if __name__ == "__main__":
    # This block defines how the script accepts arguments when called from the command line
    parser = argparse.ArgumentParser(description="Process JSON data and save model responses to a CSV file.")
    parser.add_argument("--json_path", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--custom_string", type=str, default="", help="Custom string to prepend to the input text.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the local Hugging Face model.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path for the output CSV file.")
    parser.add_argument("--method", type=str, required=True,
                        help="The processing method to use (e.g., 'nja' or 'None').")

    args = parser.parse_args()

    # Call the main function with the parsed arguments
    save_responses_to_csv(args.json_path, args.custom_string, args.model_path, args.output_csv, args.method)