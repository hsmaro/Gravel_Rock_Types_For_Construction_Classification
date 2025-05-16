import os
import random
import numpy as np
import pandas as pd
import timm
import glob

import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import warnings
warnings.filterwarnings(action='ignore')

# 내가 생성
from preprocessing import PadSquare

class Initializer:
    def __init__(self, args):
        self.args = args
        self.device = self.set_device()
        self.set_seed(args.SEED)
        self.save_path = self.prepare_features_save_dir(args.SAVE_DIR)
        self.df = self.make_dataframe()
        self.transform = self.build_transform()
        self.model = self.load_model()

    # 장치 설정
    def set_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 랜덤 시드 고정
    def set_seed(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    # 저장 장소 결정
    def prepare_features_save_dir(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        return os.path.join(save_dir, "features.npz")
    
    # 데이터 호출
    def make_dataframe(self):
        all_img_list = glob.glob('./data/train/*/*')
        df = pd.DataFrame(columns=['img_path', 'rock_type'])
        df['img_path'] = all_img_list
        df['rock_type'] = df['img_path'].apply(lambda x : str(x).split('\\')[1])
        return df

    # 이미지 데이터 전처리
    def build_transform(self):
        transform = A.Compose([
        PadSquare(value=(0, 0, 0)),
        A.Resize(self.args.IMAGE_SIZE, self.args.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2()
        ])
        return transform
    
    def load_model(self):
        model = timm.create_model(self.args.MODEL, pretrained=True)
        model.fc = torch.nn.Identity()
        model.eval().to(self.device)
        return model