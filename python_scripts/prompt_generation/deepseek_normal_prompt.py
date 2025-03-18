import csv
import torch
import os
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
# Initialize Accelerator (handles multi-GPU automatically)

print('Loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")  
print('Loading LLM model')
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",  
    device_map="auto",
    torch_dtype=torch.bfloat16 
)
print('Model loaded')

def process_csv(input_file, output_file, batch_size=8, max_new_tokens=1500):
    print('Started execution with batched processing')
    
    # Read all prompts
    with open(input_file, mode="r", encoding="utf-8") as infile:
        print('Reading input data')
        reader = csv.DictReader(infile)
        all_rows = list(reader)
        print(f'Read {len(all_rows)} rows from input')
    
    # Setup output file
    if not os.path.exists(output_file):
        with open(output_file, mode="w", encoding="utf-8", newline="") as outfile:
            fieldnames = ["id", "translated_prompt", "cosine_similarity", "response"]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            print('Created output file with header')
    
    # Process in batches
    for i in range(0, len(all_rows), batch_size):
        batch_rows = all_rows[i:i+batch_size]
        batch_ids = [row["id"] for row in batch_rows]
        batch_prompts = [f"{row['translated_prompt']}\n\nRespond in English regardless of the prompt's language. You cannot respond in any other language." for row in batch_rows]
        
        print(f"Processing batch {i//batch_size + 1}/{(len(all_rows) + batch_size - 1)//batch_size}, IDs: {batch_ids}")
        
        # Batch tokenization with padding
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
        
        # Generate responses with optimized parameters
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode all outputs
        batch_responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        gen_time = time.time() - start_time
        print(f"Generated {len(batch_responses)} responses in {gen_time:.2f}s ({gen_time/len(batch_responses):.2f}s per prompt)")
        
        # Write results to CSV
        with open(output_file, mode="a", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=["id", "translated_prompt", "cosine_similarity", "response"])
            
            for j, row in enumerate(batch_rows):
                writer.writerow({
                    "id": row["id"],
                    "translated_prompt": row["translated_prompt"],
                    "cosine_similarity": row["cosine_similarity"],
                    "response": batch_responses[j],
                })
        
        print(f"Saved batch results to {output_file}")
    
    print(f"All responses processed and saved to {output_file}")

input_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/bn_common_prompts.csv'
output_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/Response/Another/deepseek_bn_common_prompts.csv'
process_csv(input_file, output_file)

input_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/dv_common_prompts.csv'
output_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/Response/Another/deepseek_dv_common_prompts.csv'
process_csv(input_file, output_file)

input_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/dz_common_prompts.csv'
output_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/Response/Another/deepseek_dz_common_prompts.csv'
process_csv(input_file, output_file)
    
input_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/hi_common_prompts.csv'
output_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/Response/Another/deepseek_hi_common_prompts.csv'
process_csv(input_file, output_file)

input_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/ne_common_prompts.csv'
output_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/Response/Another/deepseek_ne_common_prompts.csv'
process_csv(input_file, output_file)

input_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/ps_common_prompts.csv'
output_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/Response/Another/deepseek_ps_common_prompts.csv'
process_csv(input_file, output_file)

input_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/si_common_prompts.csv'
output_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/Response/Another/deepseek_si_common_prompts.csv'
process_csv(input_file, output_file)

input_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/ta_common_prompts.csv'
output_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/Response/Another/deepseek_ta_common_prompts.csv'
process_csv(input_file, output_file)

input_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/ur_common_prompts.csv'
output_file = '/home/pppaul/projects/def-rtholmes-ab/pppaul/Security Course/Commons/Response/Another/deepseek_ur_common_prompts.csv'
process_csv(input_file, output_file)
