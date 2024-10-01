import psycopg2
from psycopg2.extras import execute_values
from config import load_config
from connect import connect


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

def insert_sentences(cursor, sentences):

    insert_sentence = '''
        INSERT INTO sentences(sentence) VALUES(%s)
    '''

    try:

        data = [(sentence, ) for sentence in sentences] # Create a list of tuples with the sentences

        execute_values(cursor, insert_sentence, data) # Execute the query with the list of tuples, more eficient than execute for each sentence

        print('Sentences inserted successfully')

    except (psycopg2.DatabaseError, Exception) as error:

        print(f"Error inserting the sentences: {error}")
        raise # Re-raise the exception

def load_sentences(file_path):

    try:

        with open(file_path, 'r') as file:
            sentences = [line.strip() for line in file]  # Read all the lines and strip the newline character
        return sentences
    
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
                    insert_sentences(cursor, sentences)
                    database_transaction.commit()

    except (psycopg2.DatabaseError, Exception) as error:

        database_transaction.rollback()
        print(f"Error: {error}")

    finally:
        if database_transaction is not None:
            database_transaction.close()
            print('Database connection closed.')

