import pandas as pd
import numpy as np
import glob
from sklearn.metrics import f1_score

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

files = glob.glob('*/y_test_pred.pkl')
for file in files:
    res = pd.read_pickle(file)

    ans_arr = np.array(list(ans['failureNum'].values))
    res_arr = np.array(list(res['failureNum'].values))

    txt = file.replace('/y_test_pred.pkl', '/result.txt')
    valid_f1 = None
    with open(txt) as t:
        valid_f1 = str(t.readline()).strip()

    print(file, valid_f1, f1_score(ans_arr, res_arr, average='macro'))
