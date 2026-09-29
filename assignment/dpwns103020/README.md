# Paper Research Agent

## 1. 프로젝트 개요

이 프로젝트는 LangChain 기반의 **논문 탐색 Agent**입니다.

사용자가 연구 주제를 입력하면 Agent가 스스로 논문 검색 Tool을 선택하고,
검색 결과 중 관련 논문을 다시 상세 조회한 뒤,
최종적으로 연구 주제와의 관련성을 정리하여 Markdown 보고서로 저장합니다.

단순히 정해진 순서대로 함수를 실행하는 것이 아니라,
LLM이 상황에 따라 Tool 사용 여부와 순서를 판단하도록 구성했습니다.

예시 연구 주제:

```text
HZO와 ZrO2 기반 MIM capacitor에서
leakage current를 줄이는 데 도움이 되는 논문 탐색