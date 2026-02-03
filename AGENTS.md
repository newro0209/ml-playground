# 작업 지침

- 주석과 문서는 한국어로 작성합니다.
- 타입 힌트는 내장 타입(`list`, `dict`, `set`, `tuple`, `str`, `int`, `float`, `bool`) 위주로 적극 활용합니다.
- Type Checking Mode가 `Basic`일 때 에러가 발생하지 않도록 작성합니다.
- Pylance(Pyright) 기준으로 항상 에러 0을 목표로 합니다. 모호한 타입은 `typing.cast`로 명시하고, `**kwargs` 전달은 피하며(필요 시 명시 인자 사용), 라이브러리 스텁 한계가 있으면 안전한 대체 호출(예: `torch.nn.Module.to(model, device)`)로 우회합니다.
- 커밋 메시지는 Gitmoji를 사용합니다. 타입을 제외한 제목과 본문 등의 내용은 한국어로 작성합니다.
