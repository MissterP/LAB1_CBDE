import chromadb
import time


def get_or_create_table(table, distance="l2", persistent=False):
    try:
        if persistent:
            client = chromadb.PersistentClient()
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
        upsert_times = []

        for i in range(0, len(sentences), segment_size):
            segment = sentences[i:i + segment_size]

            documents = segment
            metadatas = [{"text": sentence} for sentence in segment]
            ids = [str(j) for j in range(i + 1, i + len(segment) + 1)]

            try:
                start_time = time.time()
                collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
                end_time = time.time()

                upsert_times.append(end_time - start_time)

            except Exception as e:
                print(f"Error inserting in segment {i // segment_size}: {e}")

        if upsert_times:
            average_time = sum(upsert_times) / len(upsert_times)
            print(f"Average time taken to upsert: {average_time:.9f} seconds")

        print('Sentences inserted successfully')

    except Exception as e:
        print(f"Error processing segments: {e}")


def load_sentences(file_path):
    sentences = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                sentences.append(line.strip())

    except FileNotFoundError as error:
        print(f"Error loading the sentences: {error}")
        raise

    return sentences


if __name__ == '__main__':
    file_path = '../BookCorpus/sentences.txt'
    sentences = load_sentences(file_path)

    cosine = "cosine"
    l2 = "l2"

    for i in range(200, 1001, 200):
        print("Batch: " + str(i))

        print(f"Distance: {cosine}")
        collection = get_or_create_table(table="sentences", distance=cosine, persistent=False)
        insert_sentences_in_segments(collection, sentences, segment_size=i)

        print(f"Distance: {l2}")
        collection = get_or_create_table(table="sentences", distance=l2, persistent=False)
        insert_sentences_in_segments(collection, sentences, segment_size=i)

    collection = get_or_create_table(table="sentences", distance=cosine, persistent=True)
    insert_sentences_in_segments(collection, sentences)

    collection = get_or_create_table(table="sentences", distance=l2, persistent=True)
    insert_sentences_in_segments(collection, sentences)
