import os

import pandas as pd
import numpy as np
from sklearn import metrics

from passage.preprocessing import Tokenizer
from passage.layers import Embedding, GatedRecurrent, Dense
from passage.models import RNN
from passage.utils import load, save

from load import load_gender_data

trX, teX, trY, teY = load_gender_data(ntrain=10000) # Can increase up to 250K or so

# for demonstration, convert gender labels to a multiclass problem
trY = np.column_stack((trY == 0, trY == 1))
teY = np.column_stack((teY == 0, teY == 1))

tokenizer = Tokenizer(min_df=10, max_features=50000)
print trX[1] # see a blog example
trX = tokenizer.fit_transform(trX)
teX = tokenizer.transform(teX)
print tokenizer.n_features

layers = [
    Embedding(size=128, n_features=tokenizer.n_features),
    GatedRecurrent(size=256, activation='tanh', gate_activation='steeper_sigmoid', init='orthogonal', seq_output=False),
    Dense(size=2, activation='softmax', init='orthogonal') # softmax for multi-classification
]

model = RNN(layers=layers, cost='cce') # cce is classification loss for multi-category classification and softmax output
for i in range(2):
    model.fit(trX, trY, n_epochs=1)
    tr_preds = model.predict(trX[:len(teY)])
    te_preds = model.predict(teX)

    # use argmax to get the label of the high probability class
    tr_acc = metrics.accuracy_score(np.argmax(trY[:len(teY)], axis=1),
                                    np.argmax(tr_preds, axis=1))
    te_acc = metrics.accuracy_score(np.argmax(teY, axis=1),
                                    np.argmax(te_preds, axis=1))

    print i, tr_acc, te_acc

save(model, 'save_test.pkl') # How to save

model = load('save_test.pkl') # How to load

tr_preds = model.predict(trX[:len(teY)])
te_preds = model.predict(teX)

import ipdb; ipdb.set_trace();

tr_acc = metrics.accuracy_score(np.argmax(trY[:len(teY)], axis=1),
                                np.argmax(tr_preds, axis=1))
te_acc = metrics.accuracy_score(np.argmax(teY, axis=1),
                                np.argmax(te_preds, axis=1))

print tr_acc, te_acc
