import time
import psycopg2
from config import load_config
from connect import connect
from sentence_transformers import SentenceTransformer

BATCH_SIZE = 200

def extract_sentences(cursor, batch_size=500):

    select_sentences = '''
        SELECT id, sentence FROM sentences ORDER BY id    
    '''

    try:
        cursor.execute(select_sentences)
        while True:
            sentences_batch = cursor.fetchmany(batch_size) # Fetch a batch of sentences, by default 500
            if not sentences_batch:
                break
            yield sentences_batch  # Yield the batch of sentences

    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Error extracting the sentences from database: {error}")
        raise


def get_sentences(setences_tuples):

    return [sentence for _, sentence in setences_tuples]

def transform_senteces_embeddings(model, sentences):

    embeddings = model.encode(sentences)
    return embeddings

AVERAGE_TIME = []

def update_with_embeddings(cursor, setences_tuples, embeddings):
    global AVERAGE_TIME

    update_query = '''
        UPDATE sentences
        SET embedding = %s
        WHERE id = %s
    '''
    
    try:
        data = [(embedding.tolist(), id) for (id, _), embedding in zip(setences_tuples, embeddings)]
        start_time = time.time()
        cursor.executemany(update_query, data)
        end_time = time.time()
        elapsed_time = end_time - start_time
        AVERAGE_TIME.append(elapsed_time)

    except (psycopg2.DatabaseError, Exception) as error:
        print(f"Error updating the embeddings in the database: {error}")
        raise

if __name__ == '__main__':
    
    config = load_config()
    if config is None:
        print('Error: Config of the database not found')
        exit()

    database_transaction = connect(config)
    if database_transaction is None:
        print('Error: Database not found')
        exit()
    
    try:

        with database_transaction:

            with database_transaction.cursor() as fetch_cursor, database_transaction.cursor() as update_cursor:
                    
                    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

                    for sentences_tuples in extract_sentences(fetch_cursor, batch_size=BATCH_SIZE):

                        sentences = get_sentences(sentences_tuples)
                        embeddings = transform_senteces_embeddings(model, sentences)
                        update_with_embeddings(update_cursor, sentences_tuples, embeddings)
                        
                    print(f'Average time per batch: {sum(AVERAGE_TIME)/len(AVERAGE_TIME)}')
                    print('Embeddings updated successfully')
                    database_transaction.commit()

    except (psycopg2.DatabaseError, Exception) as error:

        database_transaction.rollback()
        print(f"Error: {error}")

    finally:
        if database_transaction is not None:
            database_transaction.close()
            print('Database connection closed.')

