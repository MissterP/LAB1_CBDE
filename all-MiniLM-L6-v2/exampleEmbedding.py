from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import random

ds = load_dataset("williamkgao/bookcorpus100mb") # 100MB of BookCorpus

sentences = ds['train']['text'][:10000] # take first 10k sentences

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # Load MiniLM model

embeddings = model.encode(sentences) # Compute embeddings