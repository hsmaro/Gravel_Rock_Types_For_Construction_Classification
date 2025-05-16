import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch

def change_test_path(path):
    path = str(path).split("/")
    path.insert(1, "data")
    path = "/".join(path)
    return path

def inference(model, test_dataloader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_dataloader)):
            imgs = imgs.float().to(device)
            
            pred = model(imgs)
            
            preds += pred.argmax(1).detach().cpu().numpy().tolist()
    return preds