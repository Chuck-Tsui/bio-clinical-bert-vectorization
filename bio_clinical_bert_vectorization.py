import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Load Bio-clinical BERT model and tokenizer
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Set the base directory path
base_dir = "./magic-pdf/magic-pdf"

# Function to process and vectorize text in chunks
def vectorize_text_in_chunks(text, tokenizer, model, chunk_size=512):
    tokens = tokenizer(text, return_tensors="pt", truncation=False, padding=True)
    chunks = [tokens['input_ids'][0][i:i + chunk_size] for i in range(0, len(tokens['input_ids'][0]), chunk_size)]

    embeddings_list = []
    for chunk in chunks:
        chunk_inputs = {'input_ids': chunk.unsqueeze(0)}
        with torch.no_grad():
            outputs = model(**chunk_inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings_list.append(embeddings)

    # Average all chunk vectors
    final_embedding = np.mean(embeddings_list, axis=0)
    return final_embedding

# Iterate through the folders in the base directory
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        # Look for .md files in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".md"):
                file_path = os.path.join(folder_path, file_name)
                
                # Read the contents of the .md file
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                
                # Vectorize the text in chunks
                embeddings = vectorize_text_in_chunks(text, tokenizer, model)
                
                # Save the vectorized output to a .npy file
                vector_file_path = os.path.join(folder_path, f"{file_name}_vectors.npy")
                with open(vector_file_path, 'wb') as vector_file:
                    np.save(vector_file, embeddings)

                print(f"Processed {file_name} and saved vectors to {vector_file_path}")
