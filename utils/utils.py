import os
import random
import numpy as np
import pandas as pd
from datetime import datetime
import glob

import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import warnings
warnings.filterwarnings(action='ignore')

# 내가 생성
from utils.data_utils import PadSquare

# yaml 파일 덮어쓰기
def update_config_with_args(config, args):
    for key, value in vars(args).items():
        if value is not None and key in config:
            config[key] = value
    return config

class Initializer:
    def __init__(self, info):
        self.info = info
        self.device = self.set_device()
        self.set_seed(info["SEED"])
        self.df = self.make_dataframe()
        self.train_transform, self.test_transform = self.build_transform()
        
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
        return seed
    
    # 데이터 호출
    def make_dataframe(self):
        all_img_list = glob.glob('./data/train/*/*')
        df = pd.DataFrame(columns=['img_path', 'rock_type'])
        df['img_path'] = all_img_list
        df['rock_type'] = df['img_path'].apply(lambda x : str(x).split('\\')[1])
        return df

    # 이미지 데이터 전처리
    def build_transform(self):
        # train만 여러 작업 추가 가능
        train_transform = A.Compose([
            PadSquare(value=(0, 0, 0)),
            # A.Resize(self.info["IMAGE_SIZE"], self.info["IMAGE_SIZE"]),
            A.RandomResizedCrop(size=(self.info["IMAGE_SIZE"], self.info["IMAGE_SIZE"]), scale=(0.8, 1.0), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.HueSaturationValue(p=0.2),
            A.GaussNoise(p=0.2),
            A.MotionBlur(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
            ])
        test_transform = A.Compose([
            PadSquare(value=(0, 0, 0)),
            A.Resize(self.info["IMAGE_SIZE"], self.info["IMAGE_SIZE"]),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2()
            ])
        return train_transform, test_transform

class Timer:
    def __init__(self):
        self.start_time = datetime.now()
        self.last_time = self.start_time
        print(f"[시작 시간] {self.start_time.replace(microsecond=0)}\n")

    def log(self, stage_name: str):
        now = datetime.now()
        stage_duration = now - self.last_time
        total_duration = now - self.start_time

        print(f"[{stage_name} 완료]\n")
        print(f"단계 소요 시간: {str(stage_duration).split('.')[0]}\n")
        print(f"전체 소요 시간: {str(total_duration).split('.')[0]}\n")

        self.last_time = now  # 다음 단계 대비 시간 갱신
