'''
Created on 2021年10月9日

@author: Administrator
'''
import numpy as np
drat_data_list = [
        [{"accuracy": 0.884796, "recall": 0.9400, "precision": 0.84658, "f1score":0.89085, "stage": "test"},
     {"accuracy": 0.915287, "recall": 0.92978, "precision": 0.903859, "f1score":0.9166, "stage": "dev"}],

        [{"accuracy": 0.88089, "recall": 0.942295, "precision": 0.83929, "f1score":0.88781, "stage": "test"},
     {"accuracy": 0.915287, "recall": 0.929784, "precision": 0.903859, "f1score":0.9166, "stage": "dev"}],
            [{"accuracy": 0.8834, "recall": 0.9361, "precision": 0.8469, "f1score":0.8892785, "stage": "test"},
     {"accuracy": 0.914598, "recall": 0.92451, "precision": 0.90682, "f1score":0.91557, "stage": "dev"}],
                [{"accuracy": 0.88447, "recall": 0.93482, "precision": 0.849358, "f1score":0.89004, "stage": "test"},
     {"accuracy": 0.9195862, "recall": 0.919688, "precision": 0.912984, "f1score":0.916319, "stage": "dev"}],
                    [{"accuracy": 0.890, "recall": 0.933, "precision": 0.858, "f1score":0.894, "stage": "test"},
     {"accuracy": 0.918, "recall": 0.917, "precision": 0.921, "f1score":0.918, "stage": "dev"}],
    ]
#第1/2次试验的验证集中，效果一模一样，有必然性和偶然性——确实是这样
stl_data_list = [
    [{"accuracy": 0.91264, "recall": 0.93047, "precision": 0.89871, "f1score":0.91432, "stage": "dev"},
     {"accuracy": 0.87683, "recall": 0.95059, "precision": 0.82844, "f1score":0.88531, "stage": "test"}],
        [{"accuracy": 0.91184, "recall": 0.93575, "precision": 0.89332, "f1score":0.91404, "stage": "dev"},
     {"accuracy": 0.87602, "recall": 0.95107, "precision": 0.826996, "f1score":0.88470, "stage": "test"}],
            [{"accuracy": 0.91644, "recall": 0.92772, "precision": 0.90752, "f1score":0.91750, "stage": "dev"},
     {"accuracy": 0.88366, "recall": 0.94490, "precision": 0.84185, "f1score":0.89040, "stage": "test"}],
                [{"accuracy": 0.91067, "recall": 0.92657, "precision": 0.89833, "f1score":0.91223, "stage": "dev"},
     {"accuracy": 0.87423, "recall": 0.95026, "precision": 0.82489, "f1score":0.88314, "stage": "test"}],
                    [{"accuracy": 0.91540, "recall": 0.92565, "precision": 0.90733, "f1score":0.9163966, "stage": "dev"},
     {"accuracy": 0.88390, "recall": 0.94538, "precision": 0.841922, "f1score":0.89065, "stage": "test"}],
    ]

if __name__ == '__main__':
    for name in ["accuracy", "recall", "precision", 'f1score']:
        for stage in ["test", 'dev']:
            values = []
            for data in stl_data_list:
                for a in data:
                    if a['stage']==stage:
                        values.append(a[name])
            print(name, stage, 'median', np.median(values), 'max', np.max(values))
            
            