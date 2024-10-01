from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import random

ds = load_dataset("williamkgao/bookcorpus100mb") # 100MB of BookCorpus

sentences = ds['train']['text'][:10000] # take first 10k sentences

with open('/home/pablo/Escritorio/Uni/CDBE/LAB1/all-MiniLM-L6-v2/sentences.txt', 'w') as f:
    for sentence in sentences:
        f.write(sentence + '\n')

#model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # Load MiniLM model

#embeddings = model.encode(sentences) # Compute embeddings