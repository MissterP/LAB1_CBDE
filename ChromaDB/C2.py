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


def get_top_2_similar_sentences(use_cosine=True, show_sentences=False):
    try:
        if use_cosine:
            print('Using cosine similarity')
            collection = get_or_create_table(table='sentences', distance="cosine", persistent=True)
        else:
            print('Using L2 squared distance')
            collection = get_or_create_table(table='sentences', distance="l2", persistent=True)

        results = []

        start_time = time.time()
        results = collection.query(
            query_texts=sentences_to_test,
            n_results=3,
        )
        end_time = time.time()
        final_time = (end_time - start_time)

        print(f"Time taken to load data into Chroma: {final_time:.9f} seconds")

        if show_sentences:
            for i, query in enumerate(sentences_to_test):
                print(f"Top 2 similar sentences for the sentence: {query}")

                similar_sentences = results['documents'][i][1:3]
                for result in similar_sentences:
                    print(f"- {result}")

                print("-" * 40)

    except Exception as e:
        print(f"Error in getting top similar sentences: {e}")


if __name__ == '__main__':
    for i in range(5):
        get_top_2_similar_sentences(use_cosine=True)
        get_top_2_similar_sentences(use_cosine=False)

    get_top_2_similar_sentences(use_cosine=True, show_sentences=True)
    get_top_2_similar_sentences(use_cosine=False, show_sentences=True)