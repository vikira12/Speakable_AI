"""PC에서 학습 전체 실행: 캐시 → float 학습 → INT8 → 매칭 보정 → 산출물. 국내 PC 전용 (AI Hub 데이터)

사용: python train_local.py <데이터 디렉터리> <작업 디렉터리> [--set 키=값 ...]
  예) python train_local.py D:/project/Speakable_AI/data/aihub_v01 D:/project/Speakable_AI/runs/v02
        --set model_version=speakable_v02 epochs=60 qat_epochs=3
  설정 키와 기본값은 spk_train/pipeline.py DEFAULT_CONFIG. 값은 JSON으로 읽고, 안 되면 문자열
  manifest 기본값은 manifest_verified.jsonl (prep_aihub.py의 교차 검증 통과분)
  같은 작업 디렉터리로 다시 실행하면 캐시·체크포인트를 이어서 쓴다. 로그는 <작업 디렉터리>/train.log
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spk_train.pipeline import DEFAULT_CONFIG, Run  # noqa: E402


def parse_sets(items):
    cfg = {}
    for it in items:
        k, v = it.split("=", 1)
        if k not in DEFAULT_CONFIG:
            sys.exit(f"알 수 없는 설정 키: {k}")
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    return cfg


def main(argv):
    if len(argv) < 2:
        sys.exit(__doc__)
    data_root, work = argv[0], argv[1]
    sets = argv[3:] if len(argv) > 2 and argv[2] == "--set" else []
    os.makedirs(work, exist_ok=True)
    logf = open(os.path.join(work, "train.log"), "a", encoding="utf8")

    def log(*a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    cfg = {"data_root": data_root, "work_dir": work, "manifest": "manifest_verified.jsonl"}
    cfg.update(parse_sets(sets))
    log(f"===== {time.strftime('%Y-%m-%d %H:%M:%S')} 시작: {json.dumps({k: v for k, v in cfg.items()}, ensure_ascii=False)}")
    run = Run(cfg, log=log)
    run.prepare()
    run.train()
    run.eval_float("val")
    run.quantize()
    run.calibrate()
    run.export()
    log(f"산출물: {run.out_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
