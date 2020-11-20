import pandas as pd
import numpy as np
import numpy
import glob
import os
import tensorflow as tf
from sklearn.metrics import f1_score
import keras

mapping_type = {
    'Center': 0,
    'Donut': 1,
    'Edge-Loc': 2,
    'Edge-Ring': 3,
    'Loc': 4,
    'Random': 5,
    'Scratch': 6,
    'Near-full': 7,
    'none': 8
}

ans = pd.read_pickle('data/wafer.pkl')
ans['failureNum'] = ans['answer']
ans = ans.replace({'failureNum': mapping_type})

files = glob.glob('*/*.h5')
f1_h5_pairs = []

for file in files:
    prefix = '/'.join(file.split('/')[:-1])
    result_path = os.path.join(prefix, 'result.txt')
    if not os.path.exists(result_path):
        print('Pass {}'.format(prefix))
        continue
        
    valid_f1 = None
    with open(result_path) as t:
        valid_f1 = float(str(t.readline()).strip())
        f1_h5_pairs.append((valid_f1, file))

class OneHotHelper:
  def __init__(self, labels=[]):
    self._labels = labels

  @property
  def labels(self):
    return self._labels
  
  @property
  def num_labels(self):
    return len(self._labels)

  def transform(self, normal_form):
    result = numpy.zeros((len(normal_form), self.num_labels), dtype=int)

    for row in range(len(normal_form)):
      value = normal_form[row]
      idx = self.labels.index(value)
      result[row, idx] = 1
    return result

  def recover(self, onehot_form):
    result = list()

    for row in range(len(onehot_form)):
      onehot = onehot_form[row]
      idx = numpy.argmax(onehot)
      result.append(self.labels[idx])
    return np.array(result)

with tf.device('/device:GPU:2'):
    df_valid = pd.read_pickle('data/wafer_valid_32.pkl')
    df_test = pd.read_pickle('data/wafer_test_32.pkl')

    x_valid = df_valid['waferMap'].values
    y_valid = df_valid['failureNum'].values
    x_test = df_test['waferMap'].values

    x_valid = np.array(list(x_valid), np.float32)
    x_test = np.array(list(x_test), np.float32)

    vpredict_sum = np.zeros((len(x_valid), 9))
    predict_sum = np.zeros((len(x_test), 9))
    nmodels = len(f1_h5_pairs)
    print('nmodels:', nmodels)

    folder = 'ensemble'
    i = 1
    while True:
        print(folder + str(i))
        if os.path.exists(folder + str(i)) == False:
            folder = folder + str(i)
            break
        i += 1
    os.makedirs(folder, exist_ok=True)

    i = 0
    for elem in sorted(f1_h5_pairs)[::-1][:nmodels]:
        i += 1
        print(i)
        f1, model_path = elem
        model = keras.models.load_model(model_path)
        vpredict_sum += model.predict(x_valid)
        predict_sum += model.predict(x_test)
    one_hot = OneHotHelper(labels=[0,1,2,3,4,5,6,7,8])
    vy_pred = np.array(one_hot.recover(vpredict_sum))
    y_pred = np.array(one_hot.recover(predict_sum))

    valid_f1 = f1_score(y_valid, vy_pred, average='macro')

    result_txt = os.path.join(folder, 'result.txt')

    with open(result_txt, 'w') as f:
        f.write(str(valid_f1))
        f.write('\n')
        f.write('nmodels:' + str(nmodels))

    pd.DataFrame({
        'failureNum': y_pred
    }).to_pickle(os.path.join(folder,'y_test_pred.pkl'))
    

