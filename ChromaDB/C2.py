import argparse
import time

from C0 import get_or_create_table

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


def get_top_2_similar_sentences(use_cosine=True):
    try:
        if use_cosine:
            print('Using cosine similarity')
            collection = get_or_create_table(table='sentences', distance="cosine", persistent=True)
        else:
            print('Using L2 squared distance')
            print('Using cosine similarity')
            collection = get_or_create_table(table='sentences', distance="l2", persistent=True)
            # todo pregutbar pq cuando hace el where setnece = ANY() en elotro no hace el != tipo not IN(%s)

        results = []

        for sentence in sentences_to_test:
            start_time = time.time()  # TODO aquí no hay batches? es tiempo general o entre encontar los que son más cercanos para cada frase

            result = (collection.query(
                    query_texts=[sentence],  # Chroma will embed this for you -> TODO se puede pasar todas de golpe, podríamos tenerlo en cuenta
                    n_results=3,  # how many results to return
                )
            )
            end_time = time.time()
            final_time = (end_time - start_time)

            results.append(result)
            print(f"Time taken to load data into Chroma: {final_time:.9f} seconds")

        # TODO revisar para ver si hacer esto o no
        for i, query in enumerate(sentences_to_test):
            print(f"Top 2 similar sentences for the sentence: {query}")

            # Excluir el primer resultado
            similar_sentences = results[i]['documents'][0][1:3]  # Empieza desde el segundo elemento
            for result in similar_sentences:
                print(f"- {result}")

            print("-" * 40)

    except Exception as e:
        print(f"Error in getting top similar sentences: {e}")


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

    get_top_2_similar_sentences(use_cosine=args.use_cosine)
