# os
import os
import shutil
# configuration
from ruamel.yaml import YAML
import argparse
# hadling data
import pandas as pd
from charset_normalizer import detect
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# train
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import warnings
warnings.filterwarnings(action='ignore')

# ---
from utils import loss_utils, utils, data_utils, train_utils
from models import pretrained_model
import inference

# args 설정
def set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", "--USE_MODE", type=str, required=True, help="Select Timm or Custom")
    parser.add_argument("-model", "--MODEL", type=str, required=True, help="Input collected Model Name") # inception_resnet_v2, vgg19_bn, resnet101, efficientnet_b2
    parser.add_argument("-img_size", "--IMAGE_SIZE", type=int, required=True, help="Input Image Size")
    parser.add_argument("-loss", "--CRITERION", type=str, required=False, default="ce", choices=["ce", 'weighted_ce', 'focal', 'label_smoothing'], help="Loss function to use")
    parser.add_argument("-lr", "--LEARNING_RATE", type=float, required=True, help="Input Learning Rate")
    parser.add_argument("-batch_size", "--BATCH_SIZE", type=int, required=True, help="Input Batch Size")
    parser.add_argument("-seed", "--SEED", type=int, required=False, default=41, help="SEED")
    
    return parser.parse_args()

# 캐시 삭제
def clear_all_caches_windows():
    torch_dir = os.path.join(os.environ["USERPROFILE"], ".cache", "torch")
    hf_dir = os.path.join(os.environ["USERPROFILE"], ".cache", "huggingface", "hub")

    for path in [torch_dir, hf_dir]:
        if os.path.exists(path):
            shutil.rmtree(path)
            print(f"✅ 삭제 완료: {path}")
        else:
            print(f"⚠️ 경로 없음: {path}")


def main(info):
    clear_all_caches_windows()
    timer = utils.Timer()
    # 초기화 - 장치 설정, 시드 고정, 데이터 프레임 생성, transform 생성, 사전학습 모델 업로드
    init = utils.Initializer(info)
    timer.log("초기화")
    # Split
    train, val = train_test_split(init.df, test_size=0.3, stratify=init.df['rock_type'], random_state=info["SEED"])
    
    # Encodeing
    le = LabelEncoder()
    train['rock_type'] = le.fit_transform(train['rock_type'])
    val['rock_type'] = le.transform(val['rock_type'])
    
    # Image Preprocessing
    train_dataset = data_utils.CustomDataset(train['img_path'].values, train['rock_type'].values, init.train_transform)
    train_loader = DataLoader(train_dataset, batch_size = info['BATCH_SIZE'], shuffle=True, num_workers=4,pin_memory=True,prefetch_factor=2)

    val_dataset = data_utils.CustomDataset(val['img_path'].values, val['rock_type'].values, init.test_transform)
    val_loader = DataLoader(val_dataset, batch_size=info['BATCH_SIZE'], shuffle=False, num_workers=4,pin_memory=True,prefetch_factor=2)
    timer.log("전처리")
    
    # Compile
    if info["USE_MODE"] == "Timm":
        model = pretrained_model.load_model(info["MODEL"])
    elif info["USE_MODE"] == "Custom":
        model = None
        
    optimizer = optim.Adam(model.parameters(), lr=info["LEARNING_RATE"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True) # train_utils.WarmupLR(optimizer, 1500)
    ## loss 선택
    if info["CRITERION"] == 'weighted_ce':
        criterion = loss_utils.get_weighted_cross_entropy_loss(train, "rock_type")
    elif info["CRITERION"] == 'focal':
        criterion = loss_utils.FocalLoss()
    elif info["CRITERION"] == 'label_smoothing':
        criterion = loss_utils.LabelSmoothingCrossEntropy()
    else:
        criterion = nn.CrossEntropyLoss()
    
    runs_intention = "Augmentation" # BaseModelTest
    compiler = train_utils.ModelCompiler(model, info, optimizer, scheduler, info["LEARNING_RATE"], runs_intention)
    timer.log("컴파일")
    
    # Train
    trainer = train_utils.Trainer(model, optimizer, scheduler, compiler.run_name, criterion,
                                  train_dataloaer=train_loader, val_dataloader=val_loader, max_epochs=info["EPOCHS"], device=init.device)
    infer_model = trainer.run()
    timer.log("학습")
    
    # Inference
    test = pd.read_csv('./data/test.csv')
    test["img_path"] = test["img_path"].apply(lambda x: inference.change_test_path(x))
    test_dataset = data_utils.CustomDataset(test['img_path'].values, None, init.test_transform)
    test_loader = DataLoader(test_dataset, batch_size=info['BATCH_SIZE'], shuffle=False, num_workers=0)
    preds = inference.inference(infer_model, test_dataloader=test_loader, device=init.device)
    preds = le.inverse_transform(preds)
    
    # Save submmit
    submit = pd.read_csv('./data/sample_submission.csv')
    submit['rock_type'] = preds
    if not os.path.exists(compiler.submit_path):
        os.makedirs(compiler.submit_path, exist_ok=True)
    cnt = len(os.listdir(compiler.submit_path)) + 1
    submit.to_csv(f'{compiler.submit_path}/{cnt:02d}_{info["MODEL"]}_submit.csv', index=False)
    timer.log("추론 및 저장")
    print("종료")
    
if __name__ == "__main__":
    args = set_parser()
    
    # yaml 파일 수정
    yaml = YAML()
    yaml.preserve_quotes = True  # 따옴표 유지
    with open('config.yaml', 'rb') as f:
        raw_data = f.read()
        result = detect(raw_data)
        print(f"Detected encoding: {result['encoding']}")

    with open('config.yaml', 'r', encoding=result['encoding']) as f:
        info = yaml.load(f)
    
    # 입력된 값들로 yaml 파일 덮어쓰기
    info = utils.update_config_with_args(info, args)
    
    # 덮어쓴 yaml 파일 저장
    with open("config.yaml", "w") as f:
        yaml.dump(info, f)
        
    print("program start")
    main(info)