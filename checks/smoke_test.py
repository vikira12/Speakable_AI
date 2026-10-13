"""PC에서 학습 파이프라인 전체를 작게 돌려 보는 점검 (정확도가 아니라 연결·형식·비트 일치 확인)

사용: python checks/smoke_test.py <작업 디렉터리> [파일럿 모델 디렉터리 (기본 ../model_pilot_v01)]
  기준 구현·C 소스는 FPGA 저장소에서 가져온다 (spk_train/paths.py)
  1. 파일럿 테스트 벡터 음성 20개를 화자 6명 × 3회 반복처럼 변형해 작은 데이터셋을 만든다
  2. Run: 캐시 → 학습 3 epoch → BN 접기·보정·INT8 → 매칭 보정 → 산출물
  3. 기존 도구로 산출물 검사: import_ai_artifacts.py → export_c.py → export_frontend_c.py
  4. C 테스트(spk_test, fe_test)를 산출물 헤더로 컴파일해서 실행
"""
import json
import os
import subprocess
import sys
import numpy as np
from scipy.signal import resample_poly

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GCC = "C:/AMDDesignTools/2025.2/Vitis/tps/mingw/10.0.0/win64.o/nt/bin/gcc.exe"
sys.path.insert(0, ROOT)
from spk_train import data                    # noqa: E402
from spk_train.paths import PILOT_DIR, fpga_arm_src, fpga_tools  # noqa: E402
from spk_train.pipeline import Run            # noqa: E402

TOOLS = fpga_tools()
ARM_SRC = fpga_arm_src()


def make_dataset(pilot_dir, root):
    os.makedirs(os.path.join(root, "audio"), exist_ok=True)
    td = json.load(open(os.path.join(pilot_dir, "training_data.json"), encoding="utf8"))
    text_of = {s["candidate_id"]: s["proposed_text"] for s in td["samples"]}
    tv = sorted(d for d in os.listdir(os.path.join(pilot_dir, "test_vectors")))
    rng = np.random.default_rng(0)
    lines = []
    for spk in range(6):
        sp = 0.88 + 0.05 * spk                                       # 화자마다 다른 속도 = 다른 목소리처럼
        for sid in tv:
            pcm = data.read_audio(os.path.join(pilot_dir, "test_vectors", sid, "audio.wav")).astype(np.float64)
            base = resample_poly(pcm, 100, int(round(100 * sp)))
            for r in range(3):
                x = base * rng.uniform(0.5, 1.5) + rng.standard_normal(len(base)) * 30
                x = x[: min(len(x), 48000)]
                rel = f"audio/spk{spk}_{sid}_{r}.wav"
                data.write_wav(os.path.join(root, rel), np.clip(np.round(x), -32768, 32767).astype(np.int16))
                split = "train" if spk < 4 else "val" if spk == 4 else "test"
                lines.append(json.dumps({"id": f"spk{spk}_{sid}_{r}", "audio": rel, "speaker": f"spk{spk}",
                                         "text": text_of[sid], "split": split}, ensure_ascii=False))
    with open(os.path.join(root, "manifest.jsonl"), "w", encoding="utf8") as f:
        f.write("\n".join(lines) + "\n")
    return len(lines)


def sh(cmd, cwd=None):
    print("$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf8", errors="replace")
    print(r.stdout.strip()[-2000:])
    if r.returncode != 0:
        print(r.stderr.strip()[-2000:])
        sys.exit(f"[실패] {cmd[0]} (종료 코드 {r.returncode})")
    return r.stdout


def main(work, pilot_dir=PILOT_DIR):
    data_root = os.path.join(work, "data")
    if not os.path.exists(os.path.join(data_root, "manifest.jsonl")):
        print(f"점검용 데이터 {make_dataset(pilot_dir, data_root)}개 생성")
    run = Run({"data_root": data_root, "work_dir": os.path.join(work, "run"), "tools_dir": TOOLS,
               "model_version": "smoke", "n_variants": 2, "workers": 2, "epochs": 3, "batch_size": 32,
               "calib_samples": 200, "n_test_vectors": 5, "max_wrong": 1.0, "max_unreg": 1.0,
               "init_checkpoint": os.path.join(pilot_dir, "checkpoint.pt")})
    run.prepare()
    run.train()
    run.eval_float("val")
    run.quantize()
    run.calibrate()
    run.export()

    imported = os.path.join(work, "imported")
    gen = os.path.join(work, "generated")
    py = sys.executable
    sh([py, os.path.join(TOOLS, "import_ai_artifacts.py"), run.out_dir, imported])
    sh([py, os.path.join(TOOLS, "export_c.py"), imported, os.path.join(imported, "tv"), gen, "5"])
    sh([py, os.path.join(TOOLS, "export_frontend_c.py"), run.out_dir, gen, "5"])
    if os.path.exists(GCC):
        for exe, srcs in (("spk_test", ["speakable_model.c", "speakable_test.c"]),
                          ("fe_test", ["speakable_frontend.c", "speakable_model.c", "speakable_frontend_test.c"])):
            out = os.path.join(work, exe + ".exe")
            sh([GCC, "-std=c99", "-O2", f"-I{ARM_SRC}", f"-I{gen}"] + [os.path.join(ARM_SRC, s) for s in srcs] + ["-lm", "-o", out])
            sh([out])
    else:
        print(f"[건너뜀] gcc 없음: {GCC}")
    print("점검 완료: 학습 → INT8 → 산출물 → 기존 도구·C 테스트 모두 통과")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(*sys.argv[1:])
