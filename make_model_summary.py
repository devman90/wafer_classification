import pandas as pd
import numpy as np
import numpy
import glob
import os
import tensorflow as tf
from sklearn.metrics import f1_score
import keras

model_paths = glob.glob('*/*.h5')

f1_model_pairs = []
for model_path in model_paths:
    prefix = os.path.dirname(model_path)
    result_path = os.path.join(prefix, 'result.txt')
    if not os.path.exists(result_path):
        print('Training not completed: {}'.format(prefix))
        continue

    with open(result_path) as result_file:
        f1 = float(result_file.readline().strip())
        print('Loading {}'.format(model_path))
        model = keras.models.load_model(model_path)
        f1_model_pairs.append((f1, model_path, model))

f1_model_pairs = sorted(f1_model_pairs)[::-1]

rank = 1
for f1, model_path, model in f1_model_pairs:
    print('Rank:', rank)
    rank += 1
    print('Path:', model_path)
    print('f1:', f1)
    print('Model')
    model.summary()
    print()
    print()
