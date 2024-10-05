import psycopg2
from config import load_config
from connect import connect
from sentence_transformers import SentenceTransformer

BATCH_SIZE = 500

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

def update_with_embeddings(cursor, setences_tuples, embeddings):

    update_query = '''
        UPDATE sentences
        SET embedding = %s
        WHERE id = %s
    '''
    
    try:
        data = [(embedding.tolist(), id) for (id, _), embedding in zip(setences_tuples, embeddings)]
        # Combines the id of the sentence with the embedding. 
        # We combine in a zip the list of tuples with the sentences and the list of embeddings.
        # The zip function returns a list of tuples with the elements in the same position of both lists.
        # We iterate over the zipped list and create a new list of tuples with the id of the sentence and the embedding.
        # The embedding is converted to a list of floats with the tolist() method.

        cursor.executemany(update_query, data)

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
                        
                    print('Embeddings updated successfully')
                    database_transaction.commit()

    except (psycopg2.DatabaseError, Exception) as error:

        database_transaction.rollback()
        print(f"Error: {error}")

    finally:
        if database_transaction is not None:
            database_transaction.close()
            print('Database connection closed.')

