# Language Detection with BERT

이 프로젝트는 BERT 모델을 사용하여 다언어 감지 (Language Detection)을 수행하는 코드입니다.

## 사용법
1. 프로젝트를 복제하거나 다운로드 합니다.

2. 필요한 패키지 및 라이러리를 설치합니다.
    ```bash
    pip install -r requirements.txt
    ```

3. eval.py 실행
    ```bash
    python eval.py

    # 출력 결과
    Text Processing Progress: 100%|███████████| 13000/13000 [21:19<00:00, 10.16it/s]
    Total Execution Time: 1295.24 seconds
    Match Ratio: 99.99%
    ```

## 프로젝트 구조
    ```
    .
    ├── README.md
    ├── checkpoints
    │   ├── trained_model_fold_1 # K fold 교차 검증에 사용된 각 모델
    │   ├── trained_model_fold_2 
    │   │       ...
    ├── data
    │   ├── flitto_2023_data_team_test_input_2.xlsx
    │   └── label_to_index.pkl
    ├── eval.py # 평가 
    ├── output.xlsx # 출력 결과
    ├── requirements.txt # 의존성 패키지
    └── train.py # 훈련
    ```