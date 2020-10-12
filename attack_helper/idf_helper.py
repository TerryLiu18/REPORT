import string
import nltk
import json
import math

def tokenize_words(docs):
    doc_dict = {}
    for idx, doc in enumerate(docs):
        doc_dict[idx] = doc
    with open("doc_dict.txt", "w") as doc:
        json.dump(doc_dict, doc)
    tokens = [nltk.word_tokenize(doc) for doc in docs]
    tokens = [[w for w in doc if w not in string.punctuation] for doc in tokens]
    return tokens

def build_dict(docs):
    rst = {}
    for idx,doc in enumerate(docs):
        for word in doc:
            if word not in rst.keys():
                rst[word] = [idx]
            else:
                rst[word].append(idx)
    for k, v in rst.items():
        rst[k] = list(set(v))
    rst = {k: v for k, v in sorted(rst.items(), key=lambda item:math.log(len(docs) / (1+len(item[1]))), reverse=True)}
    with open("word_dict.txt", "w") as word:
        json.dump(rst, word)

docs = tokenize_words(["you were born with potential",
"you were born with goodness and trust",
"you were born with ideals and dreams",
"you were born with greatness",
"you were born with wings",
"you are not meant for crawling, so don't",
"you have wings",
"learn to use them and fly"])
build_dict(docs)
doc_dict = json.load(open("doc_dict.txt"))
word_dict = json.load(open("word_dict.txt"))
print(word_dict)
print(doc_dict)
