import chromadb
from datasets import load_dataset
import time


def get_or_create_table(table, distance="l2", persistent=False):
    try:
        if persistent:
            client = chromadb.PersistentClient()  # hay que borrar la persistent db si quieres cambiar de distancia o usar un client normal que no guarda nada
        else:
            client = chromadb.Client()
        collection = client.get_or_create_collection(name=table + "_" + distance, metadata={"hnsw:space": distance})
        print(collection.metadata)

        return collection

    except Exception as e:
        print(f"Error creating or retrieving the collection: {e}")
        return None


def insert_sentences_in_segments(collection, sentences, segment_size=200):
    if collection is None:
        print("Cannot insert into collection: the collection is None.")
        return

    if not sentences or segment_size <= 0:
        print("The list of sentences is empty or the segment size is invalid.")
        return
    try:
        start_time = time.time() # Compute the minimum, maximum, standard deviation and average time for storing the textual data -> TODO haces los batches

        for i in range(0, len(sentences), segment_size):
            # todo quizas el start time es entre fors?
            segment = sentences[i:i + segment_size]

            documents = segment  # con los segmentos si me los añade bien
            metadatas = [{"text": sentence} for sentence in segment]
            ids = [str(j) for j in range(i + 1, i + len(segment) + 1)]

            try:
                collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

            except Exception as e:
                print(f"Error inserting in segment {i // segment_size}: {e}")

        end_time = time.time()
        final_time = (end_time - start_time) / segment_size

        print(f"Time taken to load data into Chroma: {final_time:.9f} seconds")

    except Exception as e:
        print(f"Error processing segments: {e}")


if __name__ == '__main__':
    # TODO hacer que se pase del txt
    ds = load_dataset("williamkgao/bookcorpus100mb")
    sentences = ["Sentence 1.", "Sentence 2.", "Sentence 3."]
    sentences = ds['train']['text'][:10000]

    for i in range(200, 1001, 200):
        print("Batch: " + str(i))
        collection = get_or_create_table(table="sentences", distance="cosine", persistent=False)
        insert_sentences_in_segments(collection, sentences, segment_size=i)

    collection = get_or_create_table(table="sentences", distance="cosine", persistent=True)
    insert_sentences_in_segments(collection, sentences)