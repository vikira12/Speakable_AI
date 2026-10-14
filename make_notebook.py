"""Colab 노트북 생성 스크립트. 노트북을 고칠 때는 이 파일을 고치고 다시 실행한다

사용: python make_notebook.py
  speakable_prep.ipynb   AI Hub에서 필요한 녹음만 읽어 학습 데이터(data.tar) 만들기
  speakable_train.ipynb  학습 → INT8 → 매칭 보정 → 산출물
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
산출물은 `model_pilot_v01`과 같은 형식이라 PC에서 `import_ai_artifacts.py` → `export_c.py` → `export_frontend_c.py`로 그대로 반영된다.

**런타임**: 런타임 → 런타임 유형 변경 → **T4 GPU**

**Google Drive 구성**
```
MyDrive/speakable/
  data/            ← 데이터 준비 노트북 결과: manifest.jsonl + audio/ (또는 data.tar 하나)
  work/            ← 이 노트북이 만든다: cache/ checkpoints/ artifacts/
```
`manifest.jsonl` 한 줄 = 발화 하나: `{"id", "audio"(data 기준 상대 경로, 16kHz mono), "speaker", "text", "text_norm", "split"(선택)}`
(PC의 `prep_aihub.py --tar`가 만드는 `data.tar`를 `data/`에 그대로 올리면 된다)

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


# ================================================================ 데이터 준비 노트북

md("""
# 스피커블 학습 데이터 만들기 (AI Hub 013 구음장애 음성인식, Colab)

원천 zip(27~226GB)을 통째로 받지 않고, **짧은 생활 표현·명령어 스크립트 녹음만** 골라 그 부분만 받아서
문장 구간(16kHz wav)으로 자른다. 결과는 학습 노트북이 읽는 `data.tar` 하나.

처리 방식 (PC의 `prep_aihub.py`와 같음)
- 라벨(녹음마다 전체 텍스트 하나)에서 짧은 문장 비율 70% 이상인 녹음 선택
- 에너지 VAD 구간 수 = 문장 수인 녹음만 순서대로 대응 (VAD 설정 9가지 시도)
- 같은 문장을 읽은 다른 녹음들과 구간 길이 패턴 비교(교차 검증) → 통과한 녹음만 사용
- '+', '/' 표기가 있는 문장, 3초 넘는 구간 제외

**런타임**: GPU 필요 없음 (기본 CPU 런타임). 디스크도 거의 쓰지 않는다.

**준비**
1. AI Hub 마이페이지에서 API 키 발급 → Colab 왼쪽 열쇠 아이콘(보안 비밀)에 `AIHUB_APIKEY`로 등록, 노트북 액세스 허용
2. 이 데이터셋의 이용 조건(국외 서버 처리, 재배포 금지)을 확인한다. 만든 데이터는 본인 Drive에만 둔다

**Drive 구성**
```
MyDrive/speakable/
  prep/    ← 진행 상황: chunk_*.tar(구간 묶음), manifest.jsonl, prep_report.json, labels/
  data/    ← 결과 data.tar (학습 노트북 입력)
```
세션이 끊기면 처음부터 다시 실행하면 된다. 처리한 녹음은 건너뛰고 이어서 진행한다.
""")

md("## 1. 준비")
code("""
import os, sys, subprocess, zipfile
from google.colab import drive, userdata
drive.mount('/content/drive')
""")
md(CLONE_MD)
code(CLONE_CODE)

md("""
## 2. 설정
`SOURCES`에서 처리할 원천 zip을 고른다. 예상 결과(짧은 문장 녹음 수 / 화자 / 짧은 문장 수, 라벨 기준):

| 원천 | filekey | 크기 | 짧은 문장 녹음 | 화자 | 짧은 문장 |
|---|---|---|---|---|---|
| TS01 뇌신경 | 32705 | 27GB | 62 | 49 | 약 4,900 |
| VS01 뇌신경(검증) | 65521 | 75GB | 190 | 106 | 약 20,900 |
| TS02 언어청각 | 55286 | 101GB | 203 | 142 | 약 23,500 |
| TS03 후두 | 55287 | 226GB | 656 | 378 | 약 50,700 |

실제로 받는 양은 짧은 문장 녹음 부분뿐이다 (각 원천의 약 5~15%).
""")
code("""
from spk_train import aihub, prep

AIHUB_KEY = userdata.get('AIHUB_APIKEY')
DATASET = 608
LABEL_KEYS = [31785, 31787, 551677, 65520]          # TL01, TL03, TL02, VL01
SOURCES = [
    (32705, 'TS01_뇌신경장애.zip'),
    (65521, 'VS01_뇌신경장애.zip'),
    (55286, 'TS02_언어청각장애.zip'),
    (55287, 'TS03_후두장애.zip'),
]
WORKERS = 6                                          # 동시에 받는 녹음 수
PREP_DRIVE = '/content/drive/MyDrive/speakable/prep'
DATA_TAR   = '/content/drive/MyDrive/speakable/data/data.tar'
LOCAL_OUT  = '/content/prep_out'
""")

md("""
## 3. 라벨 받기 (약 19MB)
""")
code("""
LABEL_DIR = os.path.join(PREP_DRIVE, 'labels')
label_zips = sorted(os.path.join(LABEL_DIR, f) for f in os.listdir(LABEL_DIR) if f.endswith('.zip')) if os.path.isdir(LABEL_DIR) else []
if len(label_zips) < len(LABEL_KEYS):
    label_zips = []
    for k in LABEL_KEYS:
        label_zips += aihub.download_tar(DATASET, k, AIHUB_KEY, LABEL_DIR)
recs = [r for r in prep.load_labels(label_zips) if prep.is_phrase_script(r) and r['speaker']]
print(f'라벨 zip {len(label_zips)}개, 짧은 문장 스크립트 녹음 {len(recs)}개, 화자 {len({r["speaker"] for r in recs})}명')
""")

md("""
## 4. 원격 읽기 확인
서버가 HTTP Range(일부 구간만 받기)를 지원하는지, tar 안의 part 구성을 확인한다. tar 헤더만 읽으므로 몇 MB만 받는다.
**여기서 "Range를 지원하지 않음"이 나오면 멈추고 알려 줄 것** (이 방식으로는 진행할 수 없음).
""")
code("""
new_src, segs = aihub.probe(DATASET, SOURCES[0][0], AIHUB_KEY)
""")

md("""
## 5. 구간 만들기
원천마다 짧은 문장 녹음만 받아 처리한다. 50개 녹음마다 Drive(`prep/`)에 보관한다.
""")
code("""
po = prep.PrepOutput(LOCAL_OUT)
sync = prep.DriveSync(LOCAL_OUT, PREP_DRIVE, every=50)
if not os.path.exists(po.rep_path):
    sync.restore()
    po = prep.PrepOutput(LOCAL_OUT)

for key, name in SOURCES:
    new_src, segs = aihub.probe(DATASET, key, AIHUB_KEY, log=lambda *a: None)
    base = next(b for b in segs if b.endswith(name))
    sg = segs[base]
    prep.process_remote(lambda: zipfile.ZipFile(aihub.ConcatFile(new_src(), sg)), recs, po, name,
                        workers=WORKERS, on_add=sync.added)
    sync.sync()
""")

md("""
## 6. 교차 검증 · data.tar
정렬 교차 검증을 통과한 구간만 `data.tar`로 묶어 `data/`에 저장한다.
""")
code("""
os.makedirs(os.path.dirname(DATA_TAR), exist_ok=True)
summary = po.finalize(tar_path='/content/data.tar')
subprocess.run(['cp', '/content/data.tar', DATA_TAR + '.tmp'], check=True)
os.replace(DATA_TAR + '.tmp', DATA_TAR)
for f in ('manifest_verified.jsonl', 'prep_report.json'):
    subprocess.run(['cp', os.path.join(LOCAL_OUT, f), PREP_DRIVE], check=True)
print('저장:', DATA_TAR, f'{os.path.getsize(DATA_TAR) / 1e9:.2f} GB')
""")

md("""
## 7. 들어 보며 확인
무작위 구간 20개. 소리와 문장이 맞지 않는 것이 많으면 학습 전에 알려 줄 것.
""")
code("""
import json, random
from IPython.display import Audio, display
lines = [json.loads(l) for l in open(os.path.join(LOCAL_OUT, 'manifest_verified.jsonl'), encoding='utf8')]
for l in random.Random(0).sample(lines, min(20, len(lines))):
    print(l['speaker'], '|', l['text'], '|', l['source'], l['start'], 's')
    display(Audio(os.path.join(LOCAL_OUT, l['audio'])))
""")

md("""
## 다음
학습 노트북(`speakable_train.ipynb`)의 데이터 위치가 `MyDrive/speakable/data`라서 그대로 실행하면 이 `data.tar`를 쓴다.
`label_field='text_norm'`(평서문·의문문 같은 표현)이 기본이다.
""")
save("speakable_prep.ipynb", gpu=False)
