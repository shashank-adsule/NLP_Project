import nltk
from nltk.corpus import stopwords

# make sure stopwords are available
nltk.download('stopwords', quiet=True)

class StopwordRemoval():

    def __init__(self):
        # load english stopwords from nltk
        self.stop_words = set(stopwords.words('english'))

    def fromList(self, text):

        stopwordRemovedText = []

        for sentence in text:
            filtered_sentence = []

            for token in sentence:
                if token.lower() not in self.stop_words and token.strip():
                    filtered_sentence.append(token)

            stopwordRemovedText.append(filtered_sentence)

        return stopwordRemovedText


    # corpus based stopword discovery
    def buildCorpusStopwords(self, all_docs_text, df_threshold=0.85):

        total_docs = len(all_docs_text)
        doc_freq = {}

        for doc in all_docs_text:
            words_in_doc = set()

            for sentence in doc:
                for token in sentence:
                    token = token.lower().strip()
                    if token:
                        words_in_doc.add(token)

            for word in words_in_doc:
                doc_freq[word] = doc_freq.get(word, 0) + 1

        corpus_stopwords = set()

        for word, freq in doc_freq.items():
            if freq / total_docs > df_threshold:
                corpus_stopwords.add(word)

        return corpus_stopwords


    # compare with nltk list
    def compareStopwords(self, corpus_stopwords):

        nltk_stopwords = set(stopwords.words('english'))

        overlap = nltk_stopwords & corpus_stopwords
        only_nltk = nltk_stopwords - corpus_stopwords
        only_corpus = corpus_stopwords - nltk_stopwords

        print("NLTK stopwords:", len(nltk_stopwords))
        print("Corpus stopwords:", len(corpus_stopwords))
        print("Overlap:", len(overlap))
        print("Only in NLTK:", list(only_nltk)[:10])
        print("Only in corpus:", list(only_corpus)[:10])

if __name__ == "__main__":
    from util import print_color

    sw = StopwordRemoval()

    print_color("Testing stopword removal",style=True)

    sample_text = [
        ["this", "is", "a", "simple", "test", "sentence"],
        ["the", "aircraft", "is", "flying", "in", "the", "sky"]
    ]

    cleaned = sw.fromList(sample_text)

    print_color("Original:",32)
    print(sample_text)

    print_color("After stopword removal:",32)
    print(cleaned)


    print_color("Testing corpus stopword discovery",style=True)

    corpus = [
        [["flow", "pressure", "is", "important"]],
        [["pressure", "flow", "affects", "aircraft"]],
        [["flow", "analysis", "in", "aircraft"]],
        [["pressure", "flow", "study"]]
    ]

    corpus_sw = sw.buildCorpusStopwords(corpus, df_threshold=0.6)

    print_color("Corpus stopwords:",32)
    print(corpus_sw)

    print_color("Comparing with NLTK stopwords",style=True)

    sw.compareStopwords(corpus_sw)