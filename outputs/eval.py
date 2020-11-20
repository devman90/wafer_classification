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

ans = pd.read_pickle('wafer.pkl')
ans['failureNum'] = ans['answer']
ans = ans.replace({'failureNum': mapping_type})

res = pd.read_pickle('2_test.pkl')

ans_arr = np.array(list(ans['failureNum'].values))
res_arr = np.array(list(res['failureNum'].values))

print(f1_score(ans_arr, res_arr, average='macro'))
