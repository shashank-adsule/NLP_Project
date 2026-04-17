import re
import nltk
from nltk.tokenize import sent_tokenize
import spacy

class SentenceSegmentation():

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def naive(self, text):
        # split sentences using punctuation
        segmentedText = re.split(r'[.!?]+', text)

        # remove empty strings
        segmentedText = [sent.strip() for sent in segmentedText if sent.strip() != ""]

        return segmentedText


    def punkt(self, text):
        # use nltk sentence tokenizer
        segmentedText = sent_tokenize(text)

        return segmentedText


    def spacySegmenter(self, text):
        # use spacy sentence detection
        doc = self.nlp(text)
        segmentedText = [sent.text.strip() for sent in doc.sents]

        return segmentedText


if __name__ == "__main__":
    from util import print_color

    segmenter = SentenceSegmentation()

    sample_text = """
    Dr. Smith went to Washington. He arrived at 10.30 a.m.
    The aircraft design was tested! Did the experiment succeed?
    Yes, it worked very well.
    """

    print_color("Original Text:",32)
    print(sample_text)

    print_color("Naive Segmentation",style=True)
    print(segmenter.naive(sample_text))

    print_color("NLTK Punkt Segmentation",style=True)
    print(segmenter.punkt(sample_text))

    print_color("spaCy Segmentation",style=True)
    print(segmenter.spacySegmenter(sample_text))