import os
import json
import time

# JSON helpers
def save_json(data, filepath):
    """Save python object to json file"""
    folder = os.path.dirname(filepath)

    if folder and not os.path.exists(folder):
        os.makedirs(folder)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath):
    """Load json file"""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# basic text cleaning
def clean_text(text):
    """Remove extra whitespace and newlines"""
    import re
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def flatten_sentences(sentences):
    """Convert list of sentences into one flat token list"""
    tokens = []
    for sentence in sentences:
        for token in sentence:
            tokens.append(token)
    return tokens



# vocabulary statistics
def build_vocabulary(corpus_tokens):
    """Return set of unique tokens in corpus"""
    vocab = set()

    for doc in corpus_tokens:
        for sentence in doc:
            for token in sentence:
                vocab.add(token.lower())

    return vocab


def vocab_size(corpus_tokens):
    """Number of unique tokens"""
    return len(build_vocabulary(corpus_tokens))


def token_count(corpus_tokens):
    """Total tokens in corpus"""
    count = 0

    for doc in corpus_tokens:
        for sentence in doc:
            count += len(sentence)

    return count

def top_n_tokens(corpus_tokens, n=20):
    """Return most frequent tokens"""
    from collections import Counter

    counter = Counter()

    for doc in corpus_tokens:
        for sentence in doc:
            for token in sentence:
                counter[token.lower()] += 1

    return counter.most_common(n)

# timing helper
def timer(func):
    """Simple timer decorator"""

    def wrapper(*args, **kwargs):

        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print(f"{func.__name__} finished in {end-start:.2f} seconds")

        return result

    return wrapper

# debug printing
def print_pipeline_sample(stage_name, data, n_docs=2, n_sentences=2):

    print("\nSample output for:", stage_name)

    for i, doc in enumerate(data[:n_docs]):
        print("\nDoc", i + 1)

        if isinstance(doc, list) and len(doc) > 0 and isinstance(doc[0], list):

            for j, sent in enumerate(doc[:n_sentences]):
                print("  Sentence", j + 1, ":", sent)

        else:
            for j, sent in enumerate(doc[:n_sentences]):
                print("  Sentence", j + 1, ":", sent)


def print_color(MSG,COLOR=33,style=False):
    ''' [⬛: 30 🟥:31 🟩:32 🟨:33 🟦:34 🟪:35 🩵:36 ⬜:37]'''
    if style:
        print(f"\u001b[{COLOR}m{f'| {MSG} |':=^65}\u001b[0m")
    else:
        print(f"\u001b[{COLOR}m{MSG}\u001b[0m")

if __name__=="__main__":
    text="helllo world"

    print_color(text,style=True)
    print_color(text,31)
    print_color("hi",30)
    print("hello")
