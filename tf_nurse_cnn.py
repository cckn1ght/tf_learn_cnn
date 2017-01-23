# -*- coding: utf-8 -*-
"""
Simple example using convolutional neural network to classify IMDB
sentiment dataset.

References:
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).
    - Kim Y. Convolutional Neural Networks for Sentence Classification[C]. 
    Empirical Methods in Natural Language Processing, 2014.

Links:
    - http://ai.stanford.edu/~amaas/data/sentiment/
    - http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf

"""


import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import pickle
import helper

n_words = 20000
# IMDB Dataset loading
train, valid, test = helper.load_data(path='nurse.pkl', n_words=n_words,
                                      valid_portion=0.1)

trainX, trainY = train
validX, validY = valid
testX, testY = test
# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
validX = pad_sequences(validX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
validY = to_categorical(validY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

# trainX = trainX[:20]
# trainY = trainY[:20]
# Building convolutional network
network = input_data(shape=[None, 100], name='input')
network = tflearn.embedding(network, input_dim=n_words, output_dim=128)
branch1 = conv_1d(network, 128, 3, padding='valid',
                  activation='relu', regularizer="L2")
branch2 = conv_1d(network, 128, 4, padding='valid',
                  activation='relu', regularizer="L2")
branch3 = conv_1d(network, 128, 5, padding='valid',
                  activation='relu', regularizer="L2")
network = merge([branch1, branch2, branch3], mode='concat', axis=1)
network = tf.expand_dims(network, 2)
network = global_max_pool(network)
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')
# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
# model.fit(trainX, trainY, n_epoch=100, shuffle=True,
#           show_metric=True, batch_size=32)
model.fit(trainX, trainY, n_epoch=3, shuffle=True, validation_set=(
    validX, validY), show_metric=True, batch_size=32)

# Prediction
test_pred = model.predict(testX)

tp_count = 0.
tn_count = 0.
fp_count = 0.
fn_count = 0.
total_no = 0.
total_yes = 0.

test_pred_tags = [0 if t[0] > t[1] else 1 for t in test_pred]
testY_tags = [0 if t[0] > t[1] else 1 for t in testY]

for i, p in enumerate(test_pred_tags):
    if testY_tags[i] == 0:
        total_no += 1
    if testY_tags[i] == 1:
        total_yes += 1
    if p == 0:
        if testY_tags[i] == 0:
            tn_count += 1
        else:
            fn_count += 1
    else:
        if testY_tags[i] == 1:
            tp_count += 1
        else:
            fp_count += 1


print('accuracy score: %f' % accuracy_score(testY_tags, test_pred_tags))
print('precision: %f' % precision_score(testY_tags, test_pred_tags))
print('recall: %f' % recall_score(testY_tags, test_pred_tags))
print('f1 score: %f' % f1_score(testY_tags, test_pred_tags))
print('roc area score: %f' % roc_auc_score(testY_tags, test_pred_tags))

d = {'predicted yes': pd.Series([tp_count, fp_count], index=[
                                'yes', 'no']), 'predicted no': pd.Series([fn_count, tn_count], index=['yes', 'no'])}
df = pd.DataFrame(d, columns=['predicted yes', 'predicted no'])
print('Confusion Matrix: ')
print(df)
print('------------------------------------')


def remove_unk(x):
    return [[1 if w >= n_words else w for w in sen] for sen in x]

with open('unannotated_nurse.pkl', 'rb') as f:
    raw_string, unannotated_set = pickle.load(f)

unannotated_set = remove_unk(unannotated_set)
unannotated_set = pad_sequences(unannotated_set, maxlen=100, value=0.)
unannotated_set = [[s] for s in unannotated_set]
unannotated_pred = list(map(model.predict, unannotated_set))
# unannotated_pred = model.predict(unannotated_set)

unannotated_pred_tag = ['yes' if p[0][1] > p[
    0][0] else 'no' for p in unannotated_pred]

for ss, tag in zip(raw_string, unannotated_pred_tag):
    print('Do you think this user is a nurse or no?')
    print('user bio: ' + ss)
    annotation = input().lower()
    if annotation == 'y':
        annotation = 'yes'
    if annotation == 'n':
        annotation = 'no'
    print('you entered {} and machine gussed {}'.format(annotation, tag))
    if annotation == tag:
        print('\U0001f604' + '  The machine gets it right')
    else:
        print('\U0001f47f' + '  The machine gets it wrong')
    print('')
    print('-----------------------')
    input()

# from nurse_bio_cnn_preprocess import grab_data
# f = open('nurse.dict.pkl', 'rb')
# dictionary = pickle.load(f)
# f.close()

# while True:
#     print('Please enter a nurse bio:')
#     input_bio = raw_input()
#     print('input_bio')
#     bio_words = pprs.string2words(
#         input_bio, remove_stopwords=False, stem=False)
#     bio_X = grab_data([bio_words], dictionary)
#     bio_X = remove_unk(bio_X)
#     bio_X = pad_sequences(bio_X, maxlen=100, value=0.)
#     prediction = model.predict(bio_X)[0]
#     if prediction[1] > prediction[0]:
#         pred = 'yes'
#     else:
#         pred = 'no'
#     print('Do you think this user is a nurse or not?')
#     annotation = raw_input().lower()
#     if annotation == 'y':
#         annotation = 'yes'
#     if annotation == 'n':
#         annotation = 'no'
#     if annotation == pred:
#         print(u'\U0001f604' + '  The machine gets it right')
#     else:
#         print(u'\U0001f47f' + '  The machine gets it wrong')
#     print('you entered {} and machine gussed {}'.format(annotation, pred))
#     print('-------------------------------------------------------------')
