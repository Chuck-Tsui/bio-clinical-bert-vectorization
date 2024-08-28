# Medical Text Vectorization Script

## Introduction
This script is designed to vectorize `.md` files containing medical articles using the Bio-clinical BERT model. The output is a set of `.npy` files, each representing the vectorized content of a corresponding `.md` file. This approach is particularly useful for processing long medical texts while preserving critical information for downstream tasks like classification, similarity analysis, or clustering.

## Dependencies
Ensure you have the following Python libraries installed:
- `torch`
- `transformers`
- `numpy`

You can install these dependencies using pip:

```bash
pip install torch transformers numpy
```
Directory Structure
The script expects the following directory structure:

/path/to/project
├── README.md
├── bio_clinical_bert_vectorization.py
└── magic-pdf
    ├── folder1
    │   ├── file1.md
    │   └── file1_vector.npy
    └── folder2
        ├── file2.md
        └── file2_vector.npy

README.md: This README file.
bio_clinical_bert_vectorization.py: The Python script for vectorizing .md files.
magic-pdf/: Directory containing subfolders, each with a .md file that needs to be vectorized.

Usage
1. Setting the Base Directory
Before running the script, make sure to set the base_dir variable in the script to the path of your main directory that contains the magic-pdf folder. For example:

base_dir = '.\magic-pdf\magic-pdf"

2. Running the Script
Navigate to the project directory and run the script using Python:

python bio_clinical_bert_vectorization.py

This command will process all .md files in the specified directory, generate their vectorized representations using Bio-clinical BERT, and save the resulting vectors in .npy files in the same directory as their corresponding .md files.

How It Works
Chunking and Vectorization
Chunking: The script splits each long medical text into smaller chunks, each no longer than 512 tokens (the maximum input length for BERT models).
Vectorization: Each chunk is passed through the Bio-clinical BERT model to generate a vector. The vectors from all chunks of a document are then averaged to produce a single vector representing the entire document.
Output
.npy Files: For each .md file, the script generates a corresponding .npy file. This file contains the final vectorized representation of the document, which can be used for various downstream tasks.
Notes
The script is optimized for processing long medical texts while maintaining the integrity of the information.
If your medical articles vary greatly in length, consider adjusting the max_length and stride parameters within the script for optimal results.
