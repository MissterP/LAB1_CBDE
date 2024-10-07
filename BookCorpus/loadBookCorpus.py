from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import random

ds = load_dataset("williamkgao/bookcorpus100mb") # 100MB of BookCorpus

sentences = ds['train']['text'][:10000] # take first 10k sentences

with open('./sentences.txt', 'w') as f:
    for sentence in sentences:
        f.write(sentence + '\n')

