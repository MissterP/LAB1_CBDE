import psycopg2
from config import load_config
from connect import connect

BATCH_SIZE = 500

def create_table_sentences(cursor):

    create_table = '''
        CREATE TABLE IF NOT EXISTS sentences(
            id SERIAL PRIMARY KEY,
            sentence TEXT NOT NULL,
            embedding FLOAT[] NULL
        )
    '''
    try:

        cursor.execute(create_table) 
        print('Table sentences created successfully')

    except (psycopg2.DatabaseError, Exception) as error:

        print(f"Error creating the table sentences: {error}")
        raise # Re-raise the exception

def insert_sentences(cursor, sentences, batch_size=500):

    insert_sentence = '''
        INSERT INTO sentences(sentence) VALUES(%s)
    '''

    try:
        batch = []
        for sentence in sentences:
            batch.append((sentence, )) # Create a tuple with the sentence
            if len(batch) == batch_size:
                cursor.executemany(insert_sentence, batch) # Execute the query with the batch of tuples
                batch = []

        # Insert any remaining sentences that didn't fill a complete batch
        if batch:
            cursor.executemany(insert_sentence, batch)

        print('Sentences inserted successfully')

    except (psycopg2.DatabaseError, Exception) as error:

        print(f"Error inserting the sentences: {error}")
        raise # Re-raise the exception

def load_sentences(file_path):

    try:

        with open(file_path, 'r') as file:
             for line in file:
                yield line.strip()  # Yield one sentence at a time
    
    except FileNotFoundError as error:
        print(f"Error loading the sentences: {error}")
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

            with database_transaction.cursor() as cursor:

                    create_table_sentences(cursor)
                    sentences = load_sentences('../BookCorpus/sentences.txt')
                    insert_sentences(cursor, sentences, batch_size=BATCH_SIZE)
                    database_transaction.commit()

    except (psycopg2.DatabaseError, Exception) as error:

        database_transaction.rollback()
        print(f"Error: {error}")

    finally:
        if database_transaction is not None:
            database_transaction.close()
            print('Database connection closed.')

