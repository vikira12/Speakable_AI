"""다른 저장소·폴더 위치

정수 추론·전처리의 기준 구현은 FPGA 저장소(Speakable_FPGA)의 speakable_arm_ref에 하나만 두고 참조한다.
위치는 환경 변수 SPEAKABLE_FPGA_REPO, 없으면 PC·Colab 기본 위치에서 찾는다.
"""
import os

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # speakable_ai
AI_ROOT = os.path.dirname(REPO_DIR)                                        # D:/Speakable_AI
PILOT_DIR = os.path.join(AI_ROOT, "model_pilot_v01")

_FPGA_CANDIDATES = ["D:/project/Speakable_FPGA", "/content/Speakable_FPGA"]


def fpga_repo():
    p = os.environ.get("SPEAKABLE_FPGA_REPO")
    for c in ([p] if p else []) + _FPGA_CANDIDATES:
        if os.path.exists(os.path.join(c, "speakable_arm_ref", "tools", "ref_model.py")):
            return c
    raise FileNotFoundError("FPGA 저장소(speakable_arm_ref/tools)를 찾지 못함: 환경 변수 SPEAKABLE_FPGA_REPO로 지정")


def fpga_tools():
    return os.path.join(fpga_repo(), "speakable_arm_ref", "tools")


def fpga_arm_src():
    return os.path.join(fpga_repo(), "speakable_arm_ref", "src")
