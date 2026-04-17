import re
import nltk
from nltk.tokenize import TreebankWordTokenizer
import spacy


class Tokenization():

    def __init__(self):
        self.ptb = TreebankWordTokenizer()
        self.nlp = spacy.load("en_core_web_sm")

    def naive(self, text):

        tokenizedText = []

        for sentence in text:
            tokens = re.findall(r'\b\w+\b', sentence)
            tokenizedText.append(tokens)

        return tokenizedText


    def pennTreeBank(self, text):

        tokenizedText = []

        for sentence in text:
            tokens = self.ptb.tokenize(sentence)
            tokenizedText.append(tokens)

        return tokenizedText


    def spacyTokenizer(self, text):

        tokenizedText = []

        for sentence in text:
            doc = self.nlp(sentence)
            tokens = [token.text for token in doc]
            tokenizedText.append(tokens)

        return tokenizedText


if __name__ == "__main__":
    from util import print_color

    tokenizer = Tokenization()

    sample_sentences = [
        "Dr. Smith's aircraft design isn't finished.",
        "The aircraft costs $3.5 million.",
        "NASA's engineers tested the new engine."
    ]

    print_color("Input Sentences:",32,style=True)
    for s in sample_sentences:
        print(s)

    print_color("Naive Tokenization",style=True)
    print(tokenizer.naive(sample_sentences))

    print_color("Penn Treebank Tokenization",style=True)
    print(tokenizer.pennTreeBank(sample_sentences))

    print_color("spaCy Tokenization",style=True)
    print(tokenizer.spacyTokenizer(sample_sentences))