import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ensure wordnet is available
nltk.download('wordnet', quiet=True)

class InflectionReduction:

    def porterStemmer(self, text):
        stemmer = PorterStemmer()
        reducedText = []

        for sentence in text:
            stems = [stemmer.stem(word) for word in sentence]
            reducedText.append(stems)

        return reducedText


    def wordnetLemmatizer(self, text):
        lemmatizer = WordNetLemmatizer()
        reducedText = []

        for sentence in text:
            lemmas = [lemmatizer.lemmatize(word) for word in sentence]
            reducedText.append(lemmas)

        return reducedText

    def reduce(self,text):
        return self.porterStemmer(text)

if __name__ == "__main__":
    from util import print_color

    reducer = InflectionReduction()

    sample_tokens = [
        ["running", "studies", "flying", "better"],
        ["cars", "wolves", "playing", "went"]
    ]

    print_color("\nInput Tokens:",32)
    print(sample_tokens)

    print_color("\nPorter Stemmer: ")
    print(reducer.porterStemmer(sample_tokens))

    print_color("\nWordNet Lemmatizer: ")
    print(reducer.wordnetLemmatizer(sample_tokens))