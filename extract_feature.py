import os
import numpy as np
from tqdm.auto import tqdm
import argparse

import torch
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings(action='ignore')

# 내가 생성
from feature_initializer import Initializer
from utils.data_utils import CustomDataset

# args 설정
def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-save_dir", "--SAVE_DIR", type=str, required=True, help="Input Save Path")
    parser.add_argument("-model", "--MODEL", type=str, required=True, help="Input timm Model Name")
    parser.add_argument("-img_size", "--IMAGE_SIZE", type=int, required=True, help="Input Image Size")
    parser.add_argument("-batch_size", "--BATCH_SIZE", type=int, required=True, help="Input Batch Size")
    parser.add_argument("-seed", "--SEED", type=int, required=False, default=41, help="SEED")
    
    return parser.parse_args()

def feature_extraction(args):
    init = Initializer(args) # 초기화
    
    dataset = CustomDataset(init.df['img_path'].values, init.df['rock_type'].values, init.transform)
    loader = DataLoader(dataset, batch_size = args.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)
    
    # 추출
    features = []
    labels = []
    with torch.no_grad():
        for images, rock_type in tqdm(loader):
            images = images.to(init.device)
            output = init.model(images)  # [B, 512]
            features.append(output.cpu().numpy())
            labels.extend(rock_type)
    
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)

    save_path = os.path.join(os.getcwd(), args.SAVE_DIR, "features.npz")
    np.savez(save_path, features=features, labels=labels)
    print("추출 완료")    
    
if __name__ == "__main__":
    args = set_parser()
    feature_extraction(args)