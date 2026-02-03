# ml-playground

여러 모델의 **미세조정**, **헤드 추가**, **구조 변경** 실험을 빠르게 반복하기 위한 프로젝트 골격입니다.

---

## 1. 목적과 범위

* 다양한 LM/멀티모달 모델에 대해 실험 사이클을 빠르게 구성
* 설정(YAML) 기반으로 **재현성** 확보
* 데이터/체크포인트/로그의 **명확한 분리**

---

## 2. 빠른 시작 (Quick Start)

1. 가상환경 활성화
2. 의존성 설치 (`pyproject.toml` 기준)
3. `configs/`에 실험 설정 작성
4. 학습 실행

```bash
python -m ml_playground.train --config configs/<your_exp>.yaml
```

---

## 3. SmolLM 계열 예제 (Demo + 한국어 미세조정)

SmolLM2는 본 프로젝트의 **참고 예제(reference implementation)** 입니다.
다른 모델을 실험하더라도 동일한 구조를 따르도록 설계되어 있습니다.

### 3.1 Base 데모 실행

```bash
./scripts/run_smollm2_demo.sh
```

* 기본 체크포인트: `HuggingFaceTB/SmolLM2-135M`

### 3.2 한국어 미세조정

학습용 텍스트 파일이 없으면, 기본 한국어 대화 데이터셋을 자동 다운로드하여 학습 파일을 생성합니다. (한 줄 = 한 샘플)

```bash
./scripts/run_smollm2_finetune.sh
```

#### 기본 동작

* 학습 파일: `data/processed/korean_train.txt`
* 체크포인트 저장: `checkpoints/smollm2-ko`
* 기본 데이터셋: `ohilikeit/empathetic_dialogues_mutli_turn_ko`
* RAM/VRAM 크기에 맞춰 배치 크기 자동 설정
* 기존 체크포인트 존재 시 재개 여부 질의 (`--resume-mode`로 제어 가능)

### 3.3 한국어 성능 비교

원본 모델과 미세조정 모델의 **퍼플렉시티 기반 비교**를 수행합니다.

```bash
./scripts/run_smollm2_eval.sh
```

기본 평가 파일은 `data/processed/korean_eval.txt`이며, 파일이 없으면 내장 샘플로 평가합니다.

### 3.4 GGUF 변환

`llama.cpp`의 변환 스크립트를 사용해 GGUF로 변환합니다.

```bash
LLAMA_CPP_DIR=/path/to/llama.cpp ./scripts/run_smollm2_gguf.sh
```

---

## 4. 디렉터리 구조와 책임

| 경로              | 역할               | 저장 규칙          |
| ----------------- | ------------------ | ------------------ |
| `configs/`        | 실험 설정(YAML)    | 실험 재현성의 기준 |
| `data/raw/`       | 원본 데이터        | 수정 금지          |
| `data/processed/` | 가공 데이터        | 스크립트로 생성    |
| `checkpoints/`    | 모델 가중치        | 실험별 하위 폴더   |
| `experiments/`    | 로그, 메트릭       | 실행 시 자동 생성  |
| `src/`            | 파이썬 패키지 코드 | 도메인 로직        |
| `scripts/`        | 실행 스크립트      | 반복 실행 편의     |
| `tools/`          | 보조 유틸          | 일회성/보조 작업   |
| `notebooks/`      | 분석/프로토타입    | 실험 결과 분석     |
| `tests/`          | 테스트 코드        | 회귀 방지          |
| `docs/`           | 문서               | 설계/실험 기록     |

---

## 6. 실행 규칙

* 모든 실험 산출물은 `experiments/`, `checkpoints/` 하위에만 저장
* 공통 유틸은 `src/ml_playground/utils/`에 배치
* 실행은 항상 `configs/` 기반으로 수행

---

## 7. 문제 해결 (Troubleshooting)

### `ModuleNotFoundError: No module named 'ml_playground'`

* `src` 패키지 구조이므로 다음 중 하나 필요

```bash
PYTHONPATH=./src python -m ml_playground.smollm2_demo
```

또는

```bash
pip install -e .
```

---

### `git push` 인증 오류

* GitHub CLI 인증 확인

```bash
gh auth status
```

* 미인증 시

```bash
gh auth login
```

---

## 8. 권장 실험 워크플로우

1. `configs/`에 실험 YAML 작성
2. `scripts/` 또는 `train.py`로 실행
3. 결과는 `experiments/`에서 확인
4. 의미 있는 결과만 `docs/`에 기록

---

## 9. 설계 원칙

* **헤더 기반 구조화**로 문서 탐색 비용 최소화
* **디렉터리 책임 분리**로 결합도 최소화
* **설정 중심 실행**으로 재현성 확보
