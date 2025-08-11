# 건설용 자갈 암석 종류 분류 AI 경진대회
>[**월간 데이콘 × 한국지능정보사회진흥원(NIA)**](https://dacon.io/competitions/official/236471/overview/description)
## 📌배경
- 자갈의 암석 종류는 콘크리트와 아스팔트의 강도, 내구성, 품질에 직접적인 영향을 미치기 때문에, 이를 정확히 구분하는 작업은 매우 중요합니다.
- 현재는 대부분의 품질 검사가 수작업으로 이루어지고 있어 시간과 비용이 많이 소요되며, 검사자의 숙련도에 따라 정확도 차이가 발생할 수 있습니다.
- 자갈 이미지에서 시각적 특징을 효과적으로 추출하고, 이를 바탕으로 자갈의 암석 종류를 자동으로 판별할 수 있는 모델을 설계하는 것이 목표

## 평가
- 평가 산식 : macro f1

## 일정
- 대회 기간 : 2025년 04월 8일(화) 10:00 ~ 2025년 05월 30일(금) 10:00

## 🗂데이터 소개
- 이미지 수
    - train : 380,020장
    - test : 95,006장
- 클래스 수 : 7개
    - Andesite, Basalt, Etc, Gneiss, Granite, Mud_Sandstone, Weathered_Rock
### 데이터 구성
```bash
├── train
│   ├── Andesite/
│   ├── Basalt/
│   ├── Etc/
│   ├── Gneiss/
│   ├── Granite/
│   ├── Mud_Sandstone/
│   └── Weathered_Rock/
├── test/
├── sample_submission.csv
└── tst.csv
```

## 주요 사항
- 참여 일자
    - 2025년 5월 8일 ~ 2025년 5월 30일까지 참여
- 데이터 상세
    - 암석의 종류는 기타 포함 7가지, 불균형 데이터
    ![](https://github.com/hsmaro/Gravel_Rock_Types_For_Construction_Classification/blob/main/images/label.JPG)
- 데이터 전처리
    - PadSquare 사용으로 암석 사진을 동일한 여백을 주며 정사각형으로 전처리

## 🧠모델 아키텍처 : 기본적으로 분류 성능이 좋은 대표 모델 시험
    - baseline : ```inception_resnet_v2```
    - 실험 모델 : ```efficientnet_b2``` ```resnet50d``` ```convnext_tiny```
    - Augmentation : ```RandomResizedCrop``` ```HorizontalFlip``` ```VerticalFlip``` ```RandomBrightnessContrast``` ```HueSaturationValue``` ```GaussNoise``` ```MotionBlur``` ```Normalize```
    - Loss : ```CrossEnctopyLoss``` ```Weighted CrossEntropyLoss``` ```Focal Loss``` ```Label Smoothing```
    - Optimizer : ```Adam```
    - Scheduler : ```ReduceLROnPlateau```

## 🏅성능 요약 : 상위 19%
|model|Marco-F1 (Score 기준)|
|---|---|
|inception_resnet_v2|0.72 (약 40 epochs)|
|efficientnet_b2|0.74 (약 20 epochs)|
|resnet50d|0.78 (약 40 epochs)|
|convnext_tiny|0.70 (약 40 epochs)|

## 프로젝트 구조
```bash
├── data
│   ├── train/
│   ├── test/
│   ├── sample_submission.csv
│   └── tst.csv
├── models/pretrained_model.py
├── runs/
├── submit/
├── utils/
│   ├── data_utils.py
│   ├── loss_utils.py
│   ├── train_utils.py
│   └── utils.py
├── config.yaml
├── inference.py
├── main.py
└── 
```
### 주요 디렉토리 설명
- ```models/pretrained_model.py``` : timm 모듈에 존재하는 사전학습 모델을 활용하는 코드
- ```runs/``` : 학습 과정의 가중치가 저장되는 경로
- ```utils/data_utils.py``` : CustomDataset과 PadSquare 전처리
- ```utils/loss_utils.py``` : 다양한 Loss 정의 모음
- ```utils/train_utils.py``` : compile과 학습에 관련된 파일
- ```utils/utils.py``` : 그 외 학습 과정 확인을 위한 동작 시간 등에 관련된 파일

## 접근
### 불균형 분포 속 Loss 선정 : 불균형 label 분포이기에 적합한 다양한 Loss 실험
    - CrossEntropyLoss : 대부분의 분류 문제에서 사용하는 Loss
    - Weighted_CE : weight 조정을 가한 CrossEntropyLoss로 label의 분포에 맞추어 모든 라벨이 동일한 영향을 가지도록 조정
    - Focal Loss: CrossEntropyLoss(이하: CE)를 개선하기 위해 등장했으며, 쉬운 문제에 대한 Loss를 더 낮추고, 어려운 문제에 대한 Loss 키우며 집중적으로 학습 가능
    - label_smoothing : 각 label 에 대한 확률 분포를 부드럽게 만들며, 다른 label 에 대한 확률을 높임, smoothing 조정이 가능하지만 미처 활용 못함

## 배운점&느낀점
### 데이터 증강 및 학습 전략
- 이미지 데이터이기에 Data Augmentation을 활용한 증강 전략을 보완할 필요를 느꼈습니다.
- 다양한 Loss의 사용을 통해 이미지 불균형 문제를 해결하고자 했으나 수상자의 발표 자료를 보고 모델 앙상블과 학습 방법에 대한 공부가 필요하다고 느꼈습니다.
