import wandb
import time

# 1. wandb 프로젝트 시작 (처음엔 로그인 창이 뜸)
wandb.init(
    project="wandb_test_project_1",    # 원하는 프로젝트 이름
    name="wandb_simple_test_run_1"     # 실험 이름
)

# 2. 임의의 로그 남기기
for step in range(10):
    wandb.log({
        "step": step,
        "loss": 10 - step,
        "accuracy": step / 10
    })
    time.sleep(0.5)

# 3. wandb 종료
wandb.finish()