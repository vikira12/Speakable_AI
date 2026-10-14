# Speakable AI

스피커블(구음장애 사용자용 온디바이스 음성 의사소통 보조기기)의 임베딩 CNN 학습.
결과는 `model_pilot_v01`과 같은 산출물 형식이라 FPGA 저장소(`Speakable_FPGA`)의 `speakable_arm_ref/tools`
변환·검증 도구에 그대로 들어간다. 정수 추론·전처리의 기준 구현은 FPGA 저장소에 하나만 두고 참조한다
(`spk_train/paths.py`: 환경 변수 `SPEAKABLE_FPGA_REPO`, 기본 `D:/project/Speakable_FPGA`).

> **AI Hub 데이터는 국외 다운로드·반출 금지.** 받기·자르기·학습을 모두 국내 PC에서 하고, `data/`·`runs/`
> (자른 음성, 특징 캐시, 체크포인트)를 Google Drive·Colab·GitHub 등에 올리지 않는다. 모델이 작아서 PC CPU로 학습한다.
> Colab 노트북은 팀이 직접 녹음한 데이터처럼 반출 제한이 없는 데이터에만 쓴다.

## 폴더 (`D:/project/Speakable_AI`)

```
speakable_ai/          이 저장소 (코드만)
data/                  학습 데이터 (prep_aihub.py 결과)
runs/                  캐시·체크포인트·학습 산출물
model_pilot_v01/ 등     이전 AI 팀 산출물 (파일럿 모델, 정렬 후보, 음성 점검)
```

## 저장소 구성

```
prep_aihub.py           AI Hub 원천(PC에 받은 zip 또는 원격에서 필요한 부분만) → 문장 구간 wav + manifest
train_local.py          PC 학습 전체: 캐시 → 학습 → INT8 → 매칭 보정 → 산출물 (이어서 실행 가능)
speakable_train.ipynb   Colab 학습 (반출 제한 없는 데이터 전용)
make_notebook.py        노트북 생성 스크립트 (노트북은 이 파일을 고쳐서 다시 만든다)
spk_train/
  frontend.py           학습용 전처리 (ref_frontend.py와 같은 계산, 벡터화) + 음성 증강
  data.py               manifest 읽기, 화자 단위 분할, 특징 캐시 (묶음 단위 저장)
  model.py              CNN (FPGA IP 구조 고정), AAM-softmax 헤드, BN 접기·가짜 양자화
  quant.py              활성화 보정, INT8 변환 (mult·shift), 정수 추론 시뮬레이션
  train.py              학습 루프, QAT
  evaluate.py           기기 매칭 방식 평가 (표현당 최대 5개 등록), α·T·M 보정
  export.py             산출물 디렉터리 + 테스트 벡터 (ref_model.py와 비트 일치 확인)
  pipeline.py           노트북·점검 스크립트 공용 단계 함수
  prep.py               라벨 읽기, 짧은 문장 녹음 선택, VAD 구간 분할, 정렬 교차 검증, 출력 디렉터리
  aihub.py              AI Hub 다운로드(tar 안 zip part)를 HTTP Range로 필요한 부분만 읽기
  paths.py              FPGA 저장소·파일럿 모델 위치
frontend_base/          고정 전처리 파라미터 (preprocess.json, hann, Mel 필터)
checks/
  check_quant.py        정수 추론 ↔ ref_model.py 비트 일치 + 버그 주입 3종
  smoke_test.py         파일럿 음성으로 전체 흐름 → 기존 도구 → C 테스트까지
  check_aihub_remote.py AI Hub 형식 로컬 서버로 원격 읽기 = 로컬 처리 (바이트 비교), 명령줄 --remote, 29GB zip64
```

## 데이터 준비 (AI Hub 013 구음장애 음성인식)

라벨은 녹음(수 분~수 시간)마다 전체 텍스트 하나뿐이다. 짧은 생활 표현·명령어 스크립트 녹음(짧은 문장 비율 70% 이상)만
골라 에너지 VAD로 자르고, 구간 수가 문장 수와 같은 녹음만 순서대로 대응시킨다 (VAD 설정 9가지 시도).
그다음 같은 문장을 읽은 다른 녹음들과 구간 길이 패턴을 비교해(순위 상관 ≥ 0.45, 한 칸 밀린 경우보다 높음)
통과한 녹음만 `manifest_verified.jsonl`에 넣는다. '+', '/' 표기가 있는 문장과 3초 넘는 구간은 뺀다.

원천을 통째로 받지 않는 방법 (`--remote`): AI Hub API는 tar 하나를 내려주고 큰 zip은 그 안에 `.part0, .part1, …`로
나뉘어 있다. 서버가 HTTP Range를 지원하면 tar 헤더만 읽어 part 위치를 찾고, part를 이어 붙인 가상 파일을 zipfile로 열어
짧은 문장 녹음 부분만 받는다 (원천의 약 5~15%). 디스크에는 자른 구간만 남는다.

```powershell
$env:AIHUB_APIKEY = "<AI Hub API 키>"
python prep_aihub.py D:/project/Speakable_AI/data/aihub_v01 --labels <라벨 zip 4개> --remote 32705 65521 55286 55287
```

PC에 원천 zip이 있을 때는 `--remote` 대신 `--zip <원천 zip>`. filekey: TS01 32705, VS01 65521, TS02 55286, TS03 55287.

출력: `manifest.jsonl`(정렬 전체), `manifest_verified.jsonl`, `prep_report.json`(녹음별 결과), `review.html`
(무작위 60구간 청취 확인). 같은 출력 디렉터리로 다시 실행하면 처리한 녹음은 건너뛰고(네트워크 오류 난 녹음은 다시 시도)
다른 원천과 합쳐진다.

## PC 학습

```bash
python train_local.py D:/project/Speakable_AI/data/aihub_v01 D:/project/Speakable_AI/runs/v02 --set model_version=speakable_v02 epochs=60
```

설정 키는 `spk_train/pipeline.py`의 `DEFAULT_CONFIG`. 같은 작업 디렉터리로 다시 실행하면 캐시·체크포인트를 이어서
쓴다 (`epochs`·`batch_size`·`lr`·`class_key`를 바꾸면 `checkpoints/`를 비우고 새로). 로그는 `<작업 디렉터리>/train.log`,
산출물은 `<작업 디렉터리>/artifacts/<model_version>`.

## 입력 데이터 형식

`<data_root>/manifest.jsonl`, 한 줄에 발화 하나

```json
{"id": "...", "audio": "audio/xxx.wav", "speaker": "...", "text": "문 열어?", "text_norm": "문 열어", "split": "train"}
```

- 표현 구분은 설정 `label_field` (기본 `text_norm`: 평서문·의문문을 같은 표현으로)

- audio: 16kHz mono PCM16 (wav/flac), 0.5~3초 (8,000~48,000 샘플, 범위 밖은 캐시에서 제외)
- split: 생략하면 화자 이름 해시로 train/val/test = 8/1/1. 한 화자는 한 쪽에만 들어간다
- 같은 화자가 같은 문장을 2번 이상 말한 묶음이 있어야 기기와 같은 방식(화자 안에서 등록·인식)으로 평가된다.
  없으면 화자 구분 없이 문장 단위로 평가한다 (보고서의 `mode`)

## 학습 방식

- 손실: AAM-softmax. 클래스는 `speaker_text`(화자별 문장, 기본) 또는 `text`
- 증강: 속도 0.9~1.1, 잡음 SNR 10~35dB, 잔향, 앞뒤 무음 — 발화당 원본 1 + 증강본을 캐시에 미리 저장. 학습 중 SpecAugment
- 입력 정규화: 학습 데이터 원본의 Mel 대역별 평균·표준편차 → `norm_mean.npy`, `norm_std.npy`
- INT8: BN 접기 → 활성화 99.99 백분위 보정 → 가중치 채널별 scale → mult ∈ [2^30, 2^31), shift ≤ 62. 필요하면 QAT

## 매칭 파라미터 보정

검증셋 INT8 임베딩으로 기기 점수(`α·cos(평균) + (1−α)·max cos`)를 재현해서, 아래 조건을 만족하는 T·M 중
"확실·정답 + 0.5 × 애매·후보 적중"이 가장 큰 값을 고른다. 시험셋에는 그 값을 고정해 평가만 한다.

- 등록 표현을 확실히 틀림 ≤ `max_wrong` (기본 2%)
- 미등록 음성을 확실로 오인식 ≤ `max_unreg` (기본 5%)
- 즉시 출력 기준: 각각 0.2% / 0.5% 이하, 정답률 5% 미만이면 비활성(−1)
- conflict_threshold: 같은 화자의 서로 다른 표현 대표 벡터 cosine 95백분위, adapt_min_sim: 자기 표현과의 cosine 2백분위

## PC 점검

```bash
python checks/check_quant.py
```
```bash
python checks/smoke_test.py <작업 디렉터리>
```
```bash
python checks/check_aihub_remote.py <작업 디렉터리> <원천 zip> <라벨 zip>
```

smoke_test는 정확도를 보지 않는다. 학습 → INT8 → 산출물 → `import_ai_artifacts.py` → `export_c.py` →
`export_frontend_c.py` → C 테스트(`spk_test`, `fe_test`)가 모두 통과하는지만 확인한다.

## 산출물 반영

FPGA 저장소 `speakable_arm_ref`에서:

```bash
python tools/import_ai_artifacts.py <산출물> imported/<이름>
python tools/export_c.py imported/<이름> imported/<이름>/tv generated 5
python tools/export_frontend_c.py <산출물> generated 5
```

그다음 PC C 테스트(`spk_test`, `fe_test`)와 HLS csim으로 비트 일치를 확인하고, `matching_params.json` 값을
`SPK_MATCH_PARAMS`에 반영한다.
