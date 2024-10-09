import psycopg2
import argparse
import time
from config import load_config
from connect import connect

sentences_to_test = [
    "usually , he would be tearing around the living room , playing with his toys .",
    "but just one look at a minion sent him practically catatonic .",
    "that had been megan 's plan when she got him dressed earlier .",
    "he 'd seen the movie almost by mistake , considering he was a little young for the pg cartoon , but with older cousins , along with her brothers , mason was often exposed to things that were older .",
    "she liked to think being surrounded by adults and older kids was one reason why he was a such a good talker for his age .",
    "she said .",
    "mason barely acknowledged her .",
    "instead , his baby blues remained focused on the television .",
    "since the movie was almost over , megan knew she better slip into the bedroom and finish getting ready .",
    "each time she looked into mason 's face , she was grateful that he looked nothing like his father ."
]


def get_test_sentences_embeddings(cursor):
    select_sentences = '''
        SELECT sentence, embedding FROM sentences WHERE sentence = ANY(%s)
    '''
    parameters = (sentences_to_test,)
    cursor.execute(select_sentences, parameters)
    return cursor.fetchall()


def get_all_similar_sentences(cursor, test_embeddings, use_cosine=True):
    similar_sentences = []

    for sentence_test, embedding_test in test_embeddings:
        if use_cosine:
            select_similar_sentences = '''
                SELECT sentence, embedding 
                FROM sentences 
                WHERE sentence != %s 
                ORDER BY embedding <=> %s 
                LIMIT 2
            '''
        else:
            select_similar_sentences = '''
                SELECT sentence, embedding 
                FROM sentences 
                WHERE sentence != %s 
                ORDER BY (embedding - %s) <=> 0 
                LIMIT 2
            '''

        cursor.execute(select_similar_sentences, (sentence_test, embedding_test))
        results = cursor.fetchall()
        similar_sentences.append((sentence_test, results))

    return similar_sentences


def main(use_cosine=False):
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
            with database_transaction.cursor() as cursor_test:
                sentences_test_embeddings = get_test_sentences_embeddings(cursor_test)

                # Medir tiempo para obtener frases similares
                start_time = time.time()
                similar_sentences = get_all_similar_sentences(cursor_test, sentences_test_embeddings, use_cosine)
                end_time = time.time()

                elapsed_time = end_time - start_time
                print(f"Time taken to fetch similar sentences: {elapsed_time:.4f} seconds")



            database_transaction.commit()
    except (psycopg2.DatabaseError, Exception) as error:
        database_transaction.rollback()
        print(f"Error: {error}")
    finally:
        if database_transaction is not None:
            database_transaction.close()
            print('Database connection closed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chooses the similarity metric to use')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use_cosine', action='store_true', help='Use cosine similarity if this flag is specified.')
    group.add_argument('--no_use_cosine', action='store_false', dest='use_cosine',
                       help='Use L2 squared distance if this flag is specified.')
    parser.set_defaults(use_cosine=True)
    args = parser.parse_args()

    main(use_cosine=args.use_cosine)
