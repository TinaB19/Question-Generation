import os
import gzip
import random
import math
import re
from nltk import word_tokenize

# return a list of file names end with .json.gz
def get_files(path):
    files = [pos_json for pos_json in os.listdir(path) if pos_json.endswith('.json.gz')]
    return files


def read_files(path):
    files = get_files(path)
    for file in files:
        with gzip.open(os.path.join(path, file), 'r') as g:
                for l in g:
                    yield eval(l)


def clean_html(raw):
    cleantext = re.sub('<[^>]*>', '', raw)
    text = re.sub(
        r'(?:(?:http|https):\/\/)?([-a-zA-Z0-9.]{2,256}\.[a-z]{2,4})\b(?:\/[-a-zA-Z0-9@:%_\+.~#?&//=]*)?',
        "URL", cleantext, flags=re.MULTILINE)
    return text


def read_lang(path):
    # data is a list of dictionaries, each dictionary contains answer and question as their keys
    data = list(read_files(path))

    #pairs: [ [ans0,qu0], [ans1, qu1], ..., [ansN, quN] ]
    pairs = []
    for dic in data:
        atemp = word_tokenize(clean_html(dic.get('answer')))
        qtemp = word_tokenize(clean_html(dic.get('question')))
        if 10 <= len(atemp) < 100 and 4 <= len(qtemp) < 50:
            pairs.append([' '.join(atemp), ' '.join(qtemp)])
    return pairs


def write(pairs, src_name, tgt_name):

    with open(src_name, 'w') as f:
        for pair in pairs:
            f.write(pair[0] + '\n')

    with open(tgt_name, 'w') as f:
        for pair in pairs:
            f.write(pair[1] + '\n')

def main():
    data_path = 'data/raw'
    pairs = read_lang(data_path)

    random.shuffle(pairs)
    print 'total number of data: ', len(pairs)

    train_percent = int(math.floor((80 * len(pairs)) / 100.0))
    rest = len(pairs) - train_percent
    validate_percent = rest // 2

    training_pairs = pairs[:train_percent]
    validating_pairs = pairs [train_percent: train_percent + validate_percent]
    testing_pairs = pairs[train_percent + validate_percent:]

    print 'number of training data: ', train_percent, len(training_pairs)
    print 'number of validating data: ', validate_percent, len(validating_pairs)
    print 'number of testing data: ', len(pairs) - (train_percent + validate_percent), len(testing_pairs)

    src_train = './data/processed/src-train.txt'
    tgt_train = './data/processed/tgt-train.txt'
    write(training_pairs, src_train, tgt_train)

    src_val = './data/processed/src-val.txt'
    tgt_val = './data/processed/tgt-val.txt'
    write(validating_pairs, src_val, tgt_val)

    src_test = './data/processed/src-test.txt'
    tgt_test = './data/processed/tgt-test.txt'
    write(testing_pairs, src_test, tgt_test)


if __name__ == '__main__':
    main()
