import pandas as pd
import numpy as np
import os
import subprocess
import argparse
import tqdm

experiments_dir = '/home/avic/om2/experiments'
repo_experiments_dir = '/home/avic/OOD_Orientation_Generalization/demo/exps'

def driver(exp_num, data_div):
    testing = pd.read_csv(os.path.join(experiments_dir, f'exp{exp_num}/eval/TestingFrame_Div{data_div}.csv'), index_col=0)
    predictions = pd.read_csv(os.path.join(experiments_dir, f'exp{exp_num}/eval/Div{data_div}.csv'), index_col=0)
    
    testing.drop(['Unnamed: 0.1', 'image_name', 'model_name'], axis=1, inplace=True)
    
    predictions = predictions[predictions.epoch == max(predictions.epoch)]
    predictions.drop(['data_div', 'epoch', 'image_name', 'predicted_model'], axis=1, inplace=True)
    
    final = testing.join(predictions, how='outer')
    final.object_x = final.object_x.astype(np.float64)
    final.object_y = final.object_y.astype(np.float64)
    final.object_z = final.object_z.astype(np.float64)
    
    exp_dir = os.path.join(repo_experiments_dir, f'exp{exp_num}')
    
    os.makedirs(exp_dir, exist_ok=True)
    
    final.to_csv(os.path.join(exp_dir, f'Div{data_div}.gzip'), index=False, compression='gzip')
    
                   
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('exp')
                   
    args = parser.parse_args()
    
    for data_div in tqdm.tqdm(range(10, 41, 10)):
        driver(args.exp, data_div)