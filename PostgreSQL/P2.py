import time
import psycopg2
import argparse
from config import load_config
from connect import connect
import numpy as np


sentences_to_test = ["usually , he would be tearing around the living room , playing with his toys .",
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

BATCH_SIZE = 500

def get_test_sentences_embeddings(cursor):
    select_sentences = '''
        SELECT sentence, embedding FROM sentences WHERE sentence = ANY(%s)
    '''
    parameters = (sentences_to_test,)
    cursor.execute(select_sentences, parameters)
    sentences = cursor.fetchall()
    return sentences  

def get_database_sentences_embeddings(cursor, batch_size=500):
    select_sentences = '''
        SELECT sentence, embedding FROM sentences WHERE sentence <> ALL(%s)
    '''
    parameters = (sentences_to_test,)
    cursor.execute(select_sentences, parameters)
    while True:
        sentences_batch = cursor.fetchmany(batch_size)
        if not sentences_batch:
            break
        yield sentences_batch  

FIRST_BATCH = True

def cosine_similarity(embedding_test, embedding_database):  
    
    norm_embedding_test = np.linalg.norm(embedding_test)

    norm_embedding_database = np.linalg.norm(embedding_database)

    if norm_embedding_test == 0 or norm_embedding_database == 0:
        return 0
    
    return np.dot(embedding_test, embedding_database) / (norm_embedding_test * norm_embedding_database)


def L2_squared_distance(embedding_test, embedding_database):

    return np.linalg.norm(np.array(embedding_test) - np.array(embedding_database))


def calculate_similarity(embedding_test, embedding_database, use_cosine=True):

    if len(embedding_test) != len(embedding_database):
        raise ValueError('The length of the embeddings must be the same')

    if use_cosine:
        return cosine_similarity(embedding_test, embedding_database)
    else: 
        return L2_squared_distance(embedding_test, embedding_database)
    
FIRST_TIME = True

def get_top_2_similar_sentences(sentences_test_embeddings, sentences_database_embeddings_batches, use_cosine=True):
    global FIRST_TIME

    if FIRST_TIME:
            start_time = time.time()
            FIRST_TIME = False

    top_2_similar_sentences_dict = {sentence_test: [] for sentence_test, _ in sentences_test_embeddings}

    for sentences_database_embeddings in sentences_database_embeddings_batches: 

        for sentence_test, embedding_test in sentences_test_embeddings:
            
            if embedding_test is None:
                continue

            top_2 = top_2_similar_sentences_dict[sentence_test]

            for sentence_database, embedding_database in sentences_database_embeddings:

                if embedding_database is None:
                    continue

                similarity = calculate_similarity(embedding_test, embedding_database, use_cosine)

                if len(top_2) < 2:
                    top_2.append((sentence_database, similarity))
                    if len(top_2) == 2:
                        if use_cosine:
                            top_2.sort(key=lambda x: x[1], reverse=True)
                        else:
                            top_2.sort(key=lambda x: x[1])
                else:
                    if ((use_cosine and similarity > top_2[1][1]) or
                        (not use_cosine and similarity < top_2[1][1])) and sentence_database != top_2[0][0]:
                        top_2[1] = (sentence_database, similarity)
                        if use_cosine:
                            top_2.sort(key=lambda x: x[1], reverse=True)
                        else:
                            top_2.sort(key=lambda x: x[1])

            top_2_similar_sentences_dict[sentence_test] = top_2
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Batch processed in {elapsed_time} seconds')

    for sentence_test in sentences_to_test:
        top_2 = top_2_similar_sentences_dict.get(sentence_test, [])
        print(f'\nTop 2 similar sentences for the sentence:\n"{sentence_test}"\n')
        for sentence, similarity in top_2:
            print(f'Sentence: {sentence}\nSimilarity: {similarity}\n')

def main(use_cosine=True):

    if use_cosine:
        print('Using cosine similarity')
    else:
        print('Using L2 squared distance')
    
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

            with database_transaction.cursor() as cursor_test, database_transaction.cursor() as cursor_database:

                sentences_test_embeddings = get_test_sentences_embeddings(cursor_test)
                sentences_database_embeddings = get_database_sentences_embeddings(cursor_database, batch_size=BATCH_SIZE)
                get_top_2_similar_sentences(sentences_test_embeddings, sentences_database_embeddings, use_cosine)
            
            database_transaction.commit()
                    

    except (psycopg2.DatabaseError, Exception) or (ValueError, Exception) as error:

        database_transaction.rollback()
        print(f"Error: {error}")
    

    finally:
        if database_transaction is not None:
            database_transaction.close()
            print('Database connection closed.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Chooses the similarity metric to use')

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        '--use_cosine',
        action='store_true',
        help='Use cosine similarity if this flag is specified.'
    )

    group.add_argument(
    '--no_use_cosine',
    action='store_false',
    dest='use_cosine',
    help='Use L2 squared distance if this flag is specified.'
    )

    parser.set_defaults(use_cosine=True)

    args = parser.parse_args()
    
    main(use_cosine=args.use_cosine)