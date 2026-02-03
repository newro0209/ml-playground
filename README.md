# ml-playground

여러 모델의 미세조정, 헤드 추가, 구조 변경 등 실험을 빠르게 반복하기 위한 프로젝트 골격입니다.

## 빠른 시작
1. 가상환경 활성화
2. 의존성 설치 (필요 시 `pyproject.toml` 기준)
3. `configs/`에서 실험 설정을 만들고 `src/ml_playground/train.py`로 실행

## SmolLM2 135M Base 빠른 실행
```bash
./scripts/run_smollm2_demo.sh
```

기본 체크포인트는 `HuggingFaceTB/SmolLM2-135M`입니다.

## SmolLM2 한국어 미세조정
학습용 텍스트 파일이 없으면 자동으로 일상 대화 데이터셋을 다운로드해 학습 파일을 생성합니다. (한 줄 = 한 샘플)

```bash
./scripts/run_smollm2_finetune.sh
```

기본 학습 파일은 `data/processed/korean_train.txt`이며, 학습 결과는 `checkpoints/smollm2-ko`에 저장됩니다.
기본 데이터셋은 Hugging Face의 `ohilikeit/empathetic_dialogues_mutli_turn_ko`이며, 필요하면 인자로 변경할 수 있습니다.
기본 실행은 RAM/VRAM 크기에 맞춰 배치 크기를 자동으로 설정합니다.

## Q&A
Q. `ModuleNotFoundError: No module named 'ml_playground'`가 발생해요.  
A. `src` 패키지 구조라서 편집 가능 설치 또는 `PYTHONPATH` 설정이 필요합니다.  
- 간단 실행: `PYTHONPATH=./src python -m ml_playground.smollm2_demo`  
- 영구 해결: `pip install -e .`

Q. `git push`가 인증 오류로 실패해요.  
A. GitHub CLI 인증이 되어 있다면 HTTPS 원격에서 토큰 입력 없이 push할 수 있습니다.  
- 확인: `gh auth status`  
- 미인증이면: `gh auth login` 후 다시 `git push -u origin main`

## 디렉터리 구조
- `configs/` 실험 설정(YAML)
- `data/` 데이터 저장소
- `checkpoints/` 모델 체크포인트
- `experiments/` 실험 결과(로그, 메트릭)
- `notebooks/` 분석/프로토타입
- `scripts/` 실행 스크립트
- `src/` 파이썬 패키지 코드
- `tests/` 테스트
- `docs/` 문서
- `tools/` 실험 보조 도구

## 기본 규칙
- 실험 산출물은 `experiments/`와 `checkpoints/` 아래에만 저장
- 원본 데이터는 `data/raw/`, 가공 데이터는 `data/processed/`
- 공통 유틸은 `src/ml_playground/utils/`에 배치
