import os
import numpy as np
import re
from datetime import datetime
from sklearn.metrics import f1_score

import torch
import torch.optim as optim
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 딥러닝 조기 종료
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, path='checkpoint.ckpt', delta=0, trace_func=print):
        self.patience = patience # 최소값 이후 진행 횟수
        self.verbose = verbose
        self.path = path
        self.delta = delta
        self.trace_func = trace_func
        
        self.counter = 0 # 
        self.best_score = None
        self.early_stop = False
        self.val_score = - np.Inf
        self.best_model = None
        self.best_model_state_dict = None

    def __call__(self, val_score, model): # 여러 종류 가능
        score = 1 - val_score # f1 score는 1에 가까울수록 좋다.
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        '''Saves model when validation score increase.'''
        if self.verbose:
            self.trace_func(f'Validation score increased ({self.val_score:.5f} --> {val_score:.5f}).  Saving model ...')
        
        self.best_model = model
        self.val_score = val_score
        self.best_model_state_dict = model.state_dict() # 가중치만 저장
        cnt = len(os.listdir(self.path)) + 1
        torch.save(self.best_model_state_dict, os.path.join(self.path, f"{cnt:02d}_val_loss-{self.val_score}-model.pth")) # 가중치만 저장

# Warmup Scheduler
class WarmupLR(optim.lr_scheduler.LambdaLR):

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_end_steps: int,
        last_epoch: int = -1,
    ):
        
        def wramup_fn(step: int):
            if step < warmup_end_steps:
                return float(step) / float(max(warmup_end_steps, 1))
            return 1.0
        
        super().__init__(optimizer, wramup_fn, last_epoch)

class ModelCompiler:
    def __init__(self, model, info, optimizer, scheduler, lr, runs_intention):
        self.info = info
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.run_name, self.submit_path = self.set_save_runs_path(runs_intention)
    
    # def set_optimizer(self, optimizer):
    #     optimizer = optimizer(self.model.parameters(), lr=self.lr)
    #     return optimizer
    
    # def set_scheduler(self, scheduler):
    #     scheduler = scheduler(self.optimizer, 1500)
    #     return scheduler
    
    def set_save_runs_path(self, runs_intention):
        model_category = type(self.model).__name__
        model_name = self.info["MODEL"]
        optimizer_name = type(self.optimizer).__name__
        scheduler_name = type(self.scheduler).__name__ if self.scheduler is not None else "no"
        
        runs_path = os.path.join(os.getcwd(), f"runs/{runs_intention}/{model_category}/{model_name}")
        submit_path = os.path.join(os.getcwd(), f"submit/{runs_intention}/{model_name}")
        if not os.path.exists(runs_path):
            os.makedirs(runs_path, exist_ok=True)
        
        cnt = len(os.listdir(runs_path)) + 1
        current_time = datetime.now().replace(microsecond=0).isoformat().replace(":", "_")
        runs_name = f"{cnt:02d}_{current_time}_{model_name}_{optimizer_name}_optim_{self.lr}_with_{scheduler_name}"
        runs_name = re.sub(r'[\\/:*?"<>|]', '_', runs_name)
        run_name = os.path.join(runs_path, runs_name)
        os.makedirs(run_name, exist_ok=True)
        return run_name, submit_path

class Trainer:
    def __init__(self, model, optimizer, scheduler, run_name, criterion, train_dataloaer, val_dataloader, max_epochs, device):
        self.model = model
        self.optimizer = optimizer
        self.init_scheduler = WarmupLR(self.optimizer, 10)
        self.scheduler = scheduler
        self.run_name = run_name
        self.early_stopper = EarlyStopping(patience=5, verbose=True, path=self.run_name)
        self.criterion = criterion
        self.train_dataloader = train_dataloaer
        self.val_dataloader = val_dataloader
        self.max_epoch = max_epochs # info["EPOCHS"]
        self.device = device # init.device
            
    def run(self):
        self.model.to(self.device)
        self.criterion.to(self.device)
        best_score = 0

        for epoch in range(1, self.max_epoch+1):
            self.model.train() # 훈련 상황
            train_loss = []

            for X, Y in tqdm(iter(self.train_dataloader), desc=f"Epoch: {epoch}"):
                X = X.float().to(self.device)
                Y = Y.long().to(self.device)

                # Foward
                self.optimizer.zero_grad() # 가중치 초기화

                # get prediction
                output = self.model(X)

                loss = self.criterion(output, Y)

                # back propagation
                loss.backward()

                self.optimizer.step() # 반영
                
                self.init_scheduler.step() # 초기 안정화 스케줄러 동작

                # Mini Batch 별 평가지표 저장
                train_loss.append(loss.item()) # Loss
                
            # validation 진행
            val_loss, val_score = self.validation()
            
            # 1 Epoch 후 출력
            print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}]) Val Macro F1: {val_score:.5f}')

            # schedler 반영
            if self.scheduler is not None:
                if epoch > 10:
                    self.scheduler.step(val_score)
            
            # check model early stopping point & save model if the model reached the best performance.
            self.early_stopper(val_score, self.model)
            if self.early_stopper.early_stop or epoch == self.max_epoch: # 조기종료 or 종료 시 호출되는 값들
                best_score = 1 - self.early_stopper.best_score # F1 Marco
                best_model = self.early_stopper.best_model # 모델 가중치
                
                if self.early_stopper.early_stop:
                    print("Early Stop!!")
                    print(f"Epoch: [{epoch-5}] Best Score: {best_score:.5f}") # 
                    break

                else:
                    print("Last Epoch")
                    print(f"Epoch: [{epoch}] Best Score: {best_score:.5f}") # 
        
        return best_model

    def validation(self):
        self.model.eval() # 평가 상황
        val_loss = []
        preds, true_labels = [], []
        
        with torch.no_grad(): # 가중치 업데이트 X
            for X, Y in tqdm(iter(self.val_dataloader)):
                X = X.float().to(self.device)
                Y = Y.long().to(self.device)

                pred = self.model(X)
                loss = self.criterion(pred, Y)
                
                preds += pred.argmax(1).detach().cpu().numpy().tolist()
                true_labels += Y.detach().cpu().numpy().tolist()

                # 평가지표 저장
                val_loss.append(loss.item()) # Loss
                val_score = f1_score(true_labels, preds, average='macro')
                
        return np.mean(val_loss), val_score