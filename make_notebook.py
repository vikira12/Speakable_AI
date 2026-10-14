"""Colab 학습 노트북(speakable_train.ipynb) 생성 스크립트. 노트북을 고칠 때는 이 파일을 고치고 다시 실행한다

사용: python make_notebook.py
AI Hub 데이터는 국외 반출 금지라 이 노트북에 쓸 수 없다 (PC에서 train_local.py). 팀이 직접 녹음한 데이터 등 반출 제한이 없는 데이터용
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})


def save(name, gpu):
    meta = {"kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"},
            "colab": {"provenance": []}}
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"]["gpuType"] = "T4"
    nb = {"cells": list(cells), "metadata": meta, "nbformat": 4, "nbformat_minor": 0}
    with open(os.path.join(HERE, name), "w", encoding="utf8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"{name} 생성 (셀 {len(cells)}개)")
    cells.clear()


CLONE_CODE = """
CODE_SOURCE = 'github'
GITHUB_AI   = 'vikira12/Speakable_AI'
GITHUB_FPGA = 'vikira12/Speakable_FPGA'
DRIVE_AI    = '/content/drive/MyDrive/speakable/speakable_ai'
DRIVE_FPGA  = '/content/drive/MyDrive/speakable/Speakable_FPGA'

def clone(repo, dst):
    if os.path.exists(dst):
        return
    try:
        from google.colab import userdata
        token = userdata.get('GITHUB_TOKEN')
    except Exception:
        token = None
    url = f'https://{token}@github.com/{repo}.git' if token else f'https://github.com/{repo}.git'
    r = subprocess.run(['git', 'clone', '--depth', '1', url, dst], capture_output=True, text=True)
    print(r.stderr.replace(token, '***') if token else r.stderr)

if CODE_SOURCE == 'github':
    AI_REPO, FPGA_REPO = '/content/speakable_ai', '/content/Speakable_FPGA'
    clone(GITHUB_AI, AI_REPO)
    clone(GITHUB_FPGA, FPGA_REPO)
else:
    AI_REPO, FPGA_REPO = DRIVE_AI, DRIVE_FPGA

os.environ['SPEAKABLE_FPGA_REPO'] = FPGA_REPO
sys.path.insert(0, AI_REPO)
TOOLS = os.path.join(FPGA_REPO, 'speakable_arm_ref', 'tools')
assert os.path.exists(os.path.join(AI_REPO, 'spk_train')), '학습 코드가 없음: 경로 확인'
assert os.path.exists(os.path.join(TOOLS, 'ref_model.py')), 'Speakable_FPGA/speakable_arm_ref/tools가 없음'
print('학습 코드:', AI_REPO, '/ 기준 구현:', TOOLS)
"""

CLONE_MD = """
### 코드 가져오기
저장소 두 개가 필요하다: 학습 코드 `Speakable_AI`, 정수 추론·전처리 기준 구현이 있는 `Speakable_FPGA`(`speakable_arm_ref/tools`).
- `CODE_SOURCE = 'github'`: 둘 다 clone (비공개 저장소면 Colab 보안 비밀 `GITHUB_TOKEN` 필요)
- `CODE_SOURCE = 'drive'`: Drive에 올린 두 폴더 사용
"""
md("""
# 스피커블 임베딩 모델 학습 (Colab)

FPGA IP에 맞춰 고정된 CNN(Conv16 → … → PW64 → GAP → FC64)을 학습해 **INT8 산출물**을 만든다.

> **AI Hub 데이터(구음장애 음성인식 등)는 국외 다운로드·반출 금지라 Colab에 올릴 수 없다.** 그 데이터는 PC에서
> `prep_aihub.py` → `train_local.py`로 처리한다. 이 노트북은 팀이 직접 녹음한 데이터처럼 반출 제한이 없는 데이터에만 쓴다.
산출물은 `model_pilot_v01`과 같은 형식이라 PC에서 `import_ai_artifacts.py` → `export_c.py` → `export_frontend_c.py`로 그대로 반영된다.

**런타임**: 런타임 → 런타임 유형 변경 → **T4 GPU**

**Google Drive 구성**
```
MyDrive/speakable/
  data/            ← manifest.jsonl + audio/ (또는 data.tar 하나)
  work/            ← 이 노트북이 만든다: cache/ checkpoints/ artifacts/
```
`manifest.jsonl` 한 줄 = 발화 하나: `{"id", "audio"(data 기준 상대 경로, 16kHz mono), "speaker", "text", "text_norm", "split"(선택)}`

각 단계는 Drive에 결과를 남기므로 **세션이 끊기면 처음부터 다시 실행**하면 이미 끝난 부분은 건너뛰고 이어서 진행한다.
""")

md("## 1. 준비")
code("""
import os, sys, subprocess
from google.colab import drive
drive.mount('/content/drive')
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout or 'GPU 없음 (CPU로 진행)')
print('CPU', os.cpu_count())
""")

md(CLONE_MD)
code(CLONE_CODE)


md("""
### 데이터를 로컬 디스크로 복사
Drive에서 작은 파일 수만 개를 직접 읽으면 매우 느리다. `data.tar`가 있으면 로컬에 풀고, 없으면 폴더를 복사한다.
(특징 캐시를 한 번 만들고 나면 이 단계는 필요 없다)
""")
code("""
DRIVE_DATA = '/content/drive/MyDrive/speakable/data'
WORK_DIR   = '/content/drive/MyDrive/speakable/work'
LOCAL_DATA = '/content/data'

if not os.path.exists(os.path.join(LOCAL_DATA, 'manifest.jsonl')):
    os.makedirs(LOCAL_DATA, exist_ok=True)
    tar = os.path.join(DRIVE_DATA, 'data.tar')
    if os.path.exists(tar):
        subprocess.run(['tar', '-xf', tar, '-C', LOCAL_DATA], check=True)
    else:
        subprocess.run(['cp', '-r', DRIVE_DATA + '/.', LOCAL_DATA], check=True)
print(subprocess.run(['du', '-sh', LOCAL_DATA], capture_output=True, text=True).stdout)
""")

md("""
## 2. 설정
- `label_field`: `'text_norm'` (문장부호 제외: "문 열어."와 "문 열어?"를 같은 표현으로) / `'text'` (구분)
- `class_key`: `'speaker_text'` (화자별 표현 = 기기 사용 방식과 같음) / `'text'` (화자 구분 없이 표현)
- `n_variants`: 학습 발화당 원본 1 + 증강본 수. 캐시 크기 ∝ 발화 수 × n_variants × 5KB
- `init_checkpoint`: 파일럿 가중치(`checkpoint.pt`)로 시작하려면 Drive 경로 지정
- `qat_epochs`: INT8 변환 후 정확도가 떨어지면 3~5로
- `max_wrong` / `max_unreg`: 매칭 보정 기준 (등록 표현을 확실히 틀릴 비율 / 미등록 음성을 확실로 오인식할 비율의 상한)

**학습 중 `epochs`·`batch_size`·`lr`·`class_key`를 바꾸면 이어 학습이 안 된다** (`work/checkpoints`를 비우고 새로 시작).
""")
code("""
from spk_train.pipeline import Run, DEFAULT_CONFIG

CFG = {
    'model_version': 'speakable_v02',
    'data_root': LOCAL_DATA,
    'manifest': 'manifest.jsonl',
    'label_field': 'text_norm',
    'work_dir': WORK_DIR,
    'tools_dir': TOOLS,
    'val_frac': 0.1, 'test_frac': 0.1,
    'n_variants': 4,
    'class_key': 'speaker_text',
    'epochs': 60, 'batch_size': 256, 'lr': 3e-3,
    'aam_scale': 30.0, 'aam_margin': 0.2,
    'init_checkpoint': None,
    'qat_epochs': 0,
    'max_wrong': 0.02, 'max_unreg': 0.05,
    'n_test_vectors': 20,
}
run = Run(CFG)
{k: v for k, v in run.c.items() if k != 'augment'}
""")

md("""
## 3. 데이터 목록 · 화자 분할 · 특징 캐시
CPU 작업이다 (발화 1만 개 × 증강 4 기준 수십 분). 캐시는 `work/cache`에 1000개 묶음으로 저장되어 끊겨도 이어진다.
분할 비율이나 증강 설정을 바꾸면 `work/cache`를 지우고 다시 만든다.
""")
code("""
run.prepare()
""")

md("## 4. float 학습")
code("""
run.train()
""")
code("""
import matplotlib.pyplot as plt
h = run.report['history']
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
ax[0].plot([r['epoch'] for r in h], [r['loss'] for r in h]); ax[0].set_title('train loss')
ax[1].plot([r['epoch'] for r in h], [r['train_acc'] for r in h], label='train acc')
ax[1].plot([r['epoch'] for r in h], [r['val_top1'] for r in h], label='val top-1 (enroll ≤5)')
ax[1].legend(); ax[1].set_title('accuracy'); plt.show()
rep_float = run.eval_float('val')
""")

md("""
## 5. INT8 변환
BN 접기 → 학습 데이터로 활성화 scale 보정 → 하드웨어와 같은 정수 추론으로 검증셋 평가.
float 대비 많이 떨어지면 설정에서 `qat_epochs`를 올리고 이 셀만 다시 실행한다.
""")
code("""
run.c['qat_epochs'] = CFG['qat_epochs']
run.quantize()
""")

md("""
## 6. 매칭 파라미터 보정 · 시험셋 평가
검증셋 INT8 임베딩으로 α·T·M(·즉시 출력 기준)을 정하고, 그 값을 고정해 시험셋에서 평가한다.
- **확실·오답**: 확인 화면에 틀린 표현 하나가 뜨는 경우 → 낮아야 함
- **미등록 확실**: 등록 안 한 말을 등록 표현으로 확신하는 경우 → 낮아야 함
- **애매·후보 적중**: 후보 3개 중에 정답이 있는 경우
""")
code("""
run.calibrate()
run.match_params
""")

md("## 7. 산출물 저장")
code("""
report = run.export(status='trained_not_field_validated')
import shutil
zip_path = shutil.make_archive(run.out_dir, 'zip', run.out_dir)
print('Drive 저장:', run.out_dir)
print('zip:', zip_path)
print(open(os.path.join(run.out_dir, 'MODEL_CARD.txt'), encoding='utf8').read())
""")

md("""
## 8. PC에 반영 (FPGA 저장소 `speakable_arm_ref`에서)
zip을 받아 `D:/project/Speakable_AI/<model_version>`에 푼 뒤:
```
python tools/import_ai_artifacts.py D:/project/Speakable_AI/speakable_v02 imported/speakable_v02
python tools/export_c.py imported/speakable_v02 imported/speakable_v02/tv generated 5
python tools/export_frontend_c.py D:/project/Speakable_AI/speakable_v02 generated 5
```
그다음 PC C 테스트(`spk_test`, `fe_test`)와 HLS csim으로 비트 일치를 확인하고, `matching_params.json` 값을 `SPK_MATCH_PARAMS`에 반영한다.
""")

save("speakable_train.ipynb", gpu=True)

