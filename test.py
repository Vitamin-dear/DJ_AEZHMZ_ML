import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import os
import h5py
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import time
from Medicalnet import generate_model

#测试h5数据
test_data_a = h5py.File(os.path.join('test', 'testa.h5'), 'r')['data']
test_data_b = h5py.File(os.path.join('test', 'testb.h5'), 'r')['data']
test_data = np.concatenate((np.array(test_data_a), np.array(test_data_b)))
len_test_data_a = len(np.array(test_data_a))

def all_predict(test_dataloader, loadmodel, device, result_path):
    start = time.time()
    result_df = pd.DataFrame(columns=['testa_id', 'label'])

    with torch.no_grad():
        loadmodel.to(device)
        loadmodel.eval()
        for ii, image in enumerate(test_dataloader):
            image = image.to(device)
            output = loadmodel(image)
            _, indexs = torch.max(output.data, 1)
            indexs = np.squeeze(indexs.cpu().detach().numpy()).tolist()
            if ii < len_test_data_a:
                result_df.loc[result_df.shape[0]] = [('testa_{}'.format(ii)), indexs]
            else:
                result_df.loc[result_df.shape[0]] = [('testb_{}'.format(ii - len_test_data_a)), indexs]
    result_df.to_csv(result_path, index=False)
    end = time.time()
    runing_time = end - start
    print('Test time is {:.0f}m {:.0f}s'.format(runing_time // 60, runing_time % 60))
