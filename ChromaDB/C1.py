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
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

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
        update_times = []

        for i in range(0, len(ids), segment_size):
            segment_ids = ids[i:i + segment_size]
            segment_embeddings = embeddings[i:i + segment_size]

            try:
                start_time = time.time()
                collection.update(ids=segment_ids, embeddings=segment_embeddings)
                end_time = time.time()

                update_times.append(end_time - start_time)

            except Exception as e:
                print(f"Error updating segment {i // segment_size}: {e}")

        if update_times:
            average_time = sum(update_times) / len(update_times)
            print(f"Average time taken to update: {average_time:.9f} seconds")

        print('Embeddings inserted successfully')

    except Exception as e:
        print(f"Error processing segments for update: {e}")


if __name__ == '__main__':
    collection_cosine = get_or_create_table(table='sentences', distance="cosine", persistent=True)
    sentences_cosine = extract_sentences(collection_cosine)
    embeddings_cosine = transform_sentences_embeddings(sentences_cosine)

    collection_l2 = get_or_create_table(table='sentences', distance="l2", persistent=True)
    sentences_l2 = extract_sentences(collection_l2)
    embeddings_l2 = transform_sentences_embeddings(sentences_l2)

    for i in range(200, 1001, 200):
        print("Batch: " + str(i))

        print(f"Distance: cosine")
        update_with_embeddings_in_segments(collection_cosine, sentences_cosine, embeddings_cosine, segment_size=i)

        print(f"Distance: l2")
        update_with_embeddings_in_segments(collection_l2, sentences_l2, embeddings_l2, segment_size=i)