import csv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


print('Loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it") 
print('Loading LLM model')
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16 
)
print('Model loaded')

def generate_response(input_text, max_new_tokens=1500):
        input_text = f"{input_text}\n\nRespond in English regarless of the promts language."
        print("generating token")
        input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")
        print("token generated")
        # Generate outputs
        print("Token generated, requesting for model output")
        outputs = model.generate(**input_ids, max_new_tokens=max_new_tokens)
        print("model output done")
        return tokenizer.decode(outputs[0])

def process_csv(input_file, output_file):
    print('started execution')
    # Open the input CSV file
    with open(input_file, mode="r", encoding="utf-8") as infile:
        print('start read input')
        reader = csv.DictReader(infile)
        rows = list(reader)  # Read all rows into a list
        print('read input done')
    print('read output start')
    # Open the output CSV file in append mode
    with open(output_file, mode="a", encoding="utf-8", newline="") as outfile:
        fieldnames = ["id", "translated_prompt", "response"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        # Write the header only if the file is empty
        if outfile.tell() == 0:
            writer.writeheader()
            print('writing header done.')

        # Process each row
        for row in rows:
            print('found row')
            id = row["id"]
            translated_prompt = row["Translated Prompt"]
            print(f"started processing row {id}")  # Log progress

            # Generate the response
            response = generate_response(translated_prompt)

            # Write the result to the output CSV immediately
            writer.writerow({
                "id": id,
                "translated_prompt": translated_prompt,
                "response": response,
            })

            print(f"Processed row {id}")  # Log progress

    print(f"Responses saved to {output_file}")

input_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/mixed_language_prompts.csv'
output_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/Response/gemma_mixed_language_prompts.csv'
process_csv(input_file, output_file)