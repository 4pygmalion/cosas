# cosas


# 도커 빌드
- makefile을 이용해서 도커를 빌드합니다. `{경로}`은 mlflow.pytorch.log_model로 저장된 checkpoint+직렬화된 객체를 의미합니다.

$ make build MODEL_PATH={경로}
```/bin/bash
$ make build MODEL_PATH="/vast/AI_team/mlflow_artifact/13/fe7522d516fa476fb55c003754702bd2/artifacts/model/data/model.pth"
```