from C0 import get_or_create_table
from sentence_transformers import SentenceTransformer
import time


def extract_sentences(collection):
    try:
        sentences = collection.get()
        return sentences

    except Exception as e:
        print(f"Error extracting sentences: {e}")
        return None


def transform_sentences_embeddings(sentences):
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # por defecto ya usa este modelo

        documents = sentences['documents']
        embeddings = model.encode(documents, show_progress_bar=True)

        return embeddings

    except Exception as e:
        print(f"Error transforming sentences to embeddings: {e}")
        return None


def update_with_embeddings_in_segments(collection, sentences, embeddings, segment_size=200):
    if collection is None:
        print("Cannot update collection: the collection is None.")
        return

    if sentences is None or embeddings is None:
        print("Sentences or embeddings are None, cannot proceed with update.")
        return

    ids = sentences['ids']

    if not ids or len(ids) != len(embeddings):
        print("IDs are missing or do not match the length of embeddings.")
        return

    try:
        start_time = time.time() # Compute the minimum, maximum, standard deviation and average time for storing the embeddings ->

        for i in range(0, len(ids), segment_size):
            segment_ids = ids[i:i + segment_size]
            segment_embeddings = embeddings[i:i + segment_size]

            try:
                collection.update(ids=segment_ids, embeddings=segment_embeddings)
            except Exception as e:
                print(f"Error updating segment {i // segment_size}: {e}")

        end_time = time.time()
        final_time = (end_time - start_time) / segment_size

        print(f"Time taken to load data into Chroma: {final_time:.9f} seconds")

    except Exception as e:
        print(f"Error processing segments for update: {e}")


if __name__ == '__main__':
    collection = get_or_create_table(table='sentences', distance="cosine", persistent=True)
    sentences = extract_sentences(collection)
    embeddings = transform_sentences_embeddings(sentences)

    for i in range(200, 1001, 200):
        print("Batch: " + str(i))
        update_with_embeddings_in_segments(collection, sentences, embeddings)

    update_with_embeddings_in_segments(collection, sentences, embeddings)
