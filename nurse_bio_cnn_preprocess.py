
"""
This script is what created the dataset pickled.

1) You need to download this file and put it in the same directory as this file.
https://github.com/moses-smt/mosesdecoder/raw/master/scripts/tokenizer/tokenizer.perl . Give it execution permission.

2) Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and extract it in the current directory.

3) Then run this script.
"""


import numpy
import pickle as pkl

from collections import OrderedDict

import glob
import os
import pprs
from pymongo import MongoClient
import logging
from collections import Counter
from datetime import datetime

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# tokenizer.perl is from Moses:
# https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer

client = MongoClient('localhost', 27017)
db = client.social_mining
collection = db.annotated_nurses
all_nurse_collection = client.twarcTweets.nurse_users_English_only


def load_data(has_test=True, test_percent=0.3):
    # annotated_user_ids = set()
    train_yes_words = []
    train_no_words = []
    test_yes_words = []
    test_no_words = []
    logging.info('loading training data')
    test_count = int(collection.find().count() * 0.3)
    for number, nurse in enumerate(collection.find()):
        bio_string = nurse['bio']
        # annotated_user_ids.add(nurse['user_id'])
        bio_words = pprs.string2words(
            bio_string, remove_stopwords=False, stem=False)
        class_type = nurse['class'].lower()
        if has_test:
            if class_type == 'yes':
                if number < test_count:
                    test_yes_words.append(bio_words)
                else:
                    train_yes_words.append(bio_words)
            if class_type == 'no':
                if number < test_count:
                    test_no_words.append(bio_words)
                else:
                    train_no_words.append(bio_words)
        else:
            if class_type == 'yes':
                train_yes_words.append(bio_words)
            if class_type == 'no':
                train_no_words.append(bio_words)
    if has_test:
        return train_yes_words, train_no_words, test_yes_words, test_no_words
    else:
        return train_yes_words, train_no_words


def load_unannotated_data(low_weekly_activity=1, high_weekly_activity=100):
    unannotated_data = []
    raw_string = []
    all_bio_words = []
    collected_date = datetime(2016, 3, 2)
    for i, nurse in enumerate(all_nurse_collection.find()):
        sc = nurse['statuses_count']
        dt = datetime.strptime(
            nurse['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
        delta = collected_date - dt
        spw = float(sc) * 7 / delta.days
        bio_string = nurse['description']
        bio_words = pprs.string2words(
            bio_string, remove_stopwords=False, stem=False)
        all_bio_words.append(bio_words)
        if spw < high_weekly_activity and spw > low_weekly_activity:
            raw_string.append(bio_string)
            unannotated_data.append(bio_words)

    return raw_string, unannotated_data, all_bio_words


def build_dict(sentences):

    print('Building dictionary..')
    all_words = []
    for words in sentences:
        all_words.extend(words)
    words_counter = Counter(all_words)
    sorted_counter = words_counter.most_common()
    word_dict = dict()
    for idx, tup in enumerate(sorted_counter):
        word, _ = tup
        word_dict[word] = idx + 2  # leave 0 and 1 (UNK)

    print(len(all_words), ' total words ', len(words_counter), ' unique words')
    return word_dict


def grab_data(sentences, dictionary):

    seqs = [None] * len(sentences)
    for idx, words in enumerate(sentences):
        # words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    return seqs


def main():
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    train_yes_words, train_no_words, test_yes_words, test_no_words = load_data()
    unannotated_raw, unannotated_words, all_bio_words = load_unannotated_data()

    dictionary = build_dict(all_bio_words)

    train_x_yes = grab_data(train_yes_words, dictionary)
    train_x_no = grab_data(train_no_words, dictionary)
    train_x = train_x_yes + train_x_no
    train_y = [1] * len(train_x_yes) + [0] * len(train_x_no)

    test_x_yes = grab_data(test_yes_words, dictionary)
    test_x_no = grab_data(test_no_words, dictionary)
    test_x = test_x_yes + test_x_no
    test_y = [1] * len(test_x_yes) + [0] * len(test_x_no)

    unannotated = grab_data(unannotated_words, dictionary)
    f = open('nurse.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    f.close()

    f = open('nurse.dict.pkl', 'wb')
    pkl.dump(dictionary, f, -1)
    f.close()

    f = open('unannotated_nurse.pkl', 'wb')
    pkl.dump((unannotated_raw, unannotated), f, -1)
    f.close

if __name__ == '__main__':
    main()
