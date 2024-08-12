# cosas
- https://cosas.grand-challenge.org/
Task 2: Cross-Scanner Adenocarcinoma Segmentation
Task 2 focuses on evaluating the generalisation capabilities of machine learning algorithms in adenocarcinoma segmentation across diverse whole slide image scanners. The dataset comprises image patches extracted from whole slide image scans of invasive breast carcinoma tissues, acquired with six different scanners of different manufacturers. 



# 도커 빌드
- Makefile을 이용해서 도커를 빌드합니다. `{경로}`은 mlflow.pytorch.log_model로 저장된 checkpoint+직렬화된 객체를 의미합니다.

$ make build MODEL_PATH={경로}
```bash
$ make build MODEL_PATH="/vast/AI_team/mlflow_artifact/13/fe7522d516fa476fb55c003754702bd2/artifacts/model/data/model.pth"
```

# 도커 테스트
```bash
$ make test_run
```