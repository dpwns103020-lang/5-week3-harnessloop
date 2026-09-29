import os
import time
from pathlib import Path
from urllib.parse import quote
from typing import TypedDict

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

# ============================================================
# 기본 설정
# ============================================================

OPENALEX_SEARCH_URL = "https://api.openalex.org/works"
CROSSREF_SEARCH_URL = "https://api.crossref.org/works"

REQUEST_TIMEOUT = 15


# ============================================================
# OpenAlex 검색
# ============================================================

def search_openalex(query: str, limit: int = 5) -> list[dict] | None:
    """
    OpenAlex에서 논문을 검색합니다.

    성공하면 논문 목록을 반환하고,
    요청 제한이나 오류가 발생하면 None을 반환합니다.
    """

    params = {
        "search": query,
        "per-page": limit,
    }

    max_retries = 2

    for attempt in range(max_retries + 1):

        try:
            response = requests.get(
                OPENALEX_SEARCH_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 429:

                if attempt < max_retries:
                    wait_seconds = 2 ** attempt

                    print(
                        f"[RETRY] OpenAlex rate limit. "
                        f"{wait_seconds}초 후 재시도합니다."
                    )

                    time.sleep(wait_seconds)
                    continue

                print(
                    "[FALLBACK] OpenAlex 요청 한도에 도달했습니다. "
                    "Crossref 검색으로 전환합니다."
                )

                return None

            response.raise_for_status()

        except requests.exceptions.Timeout:

            print(
                "[FALLBACK] OpenAlex 요청 시간이 초과되었습니다. "
                "Crossref 검색으로 전환합니다."
            )

            return None

        except requests.exceptions.RequestException as error:

            print(
                f"[FALLBACK] OpenAlex 검색 오류: {error}"
            )

            return None

        data = response.json()
        papers = data.get("results", [])

        results = []

        for paper in papers:

            authorships = paper.get("authorships", [])

            authors = ", ".join(
                item.get("author", {}).get("display_name", "")
                for item in authorships
                if item.get("author")
            )

            results.append(
                {
                    "source": "openalex",
                    "paper_id": paper.get("id", ""),
                    "title": paper.get("display_name", ""),
                    "authors": authors,
                    "year": paper.get("publication_year", ""),
                    "url": (
                        paper.get("doi")
                        or paper.get("id", "")
                    ),
                }
            )

        return results

    return None


# ============================================================
# Crossref 보조 함수
# ============================================================

def get_crossref_year(paper: dict) -> str:
    """
    Crossref 논문 데이터에서 출판 연도를 추출합니다.
    """

    for field in (
        "published-print",
        "published-online",
        "published",
        "issued",
    ):
        date_info = paper.get(field, {})
        date_parts = date_info.get("date-parts", [])

        if date_parts and date_parts[0]:
            return str(date_parts[0][0])

    return ""


def get_crossref_authors(paper: dict) -> str:
    """
    Crossref author 정보를 문자열로 변환합니다.
    """

    authors = []

    for author in paper.get("author", []):

        given = author.get("given", "")
        family = author.get("family", "")

        full_name = f"{given} {family}".strip()

        if full_name:
            authors.append(full_name)

    return ", ".join(authors)


# ============================================================
# Crossref 검색
# ============================================================

def search_crossref(query: str, limit: int = 5) -> list[dict]:
    """
    Crossref에서 논문을 검색합니다.
    """

    params = {
        "query": query,
        "rows": limit,
    }

    try:
        response = requests.get(
            CROSSREF_SEARCH_URL,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.exceptions.Timeout:
        return []

    except requests.exceptions.RequestException:
        return []

    data = response.json()

    papers = (
        data
        .get("message", {})
        .get("items", [])
    )

    results = []

    for paper in papers:

        title_list = paper.get("title", [])

        title = (
            title_list[0]
            if title_list
            else ""
        )

        doi = paper.get("DOI", "")

        url = (
            f"https://doi.org/{doi}"
            if doi
            else paper.get("URL", "")
        )

        results.append(
            {
                "source": "crossref",
                "paper_id": doi,
                "title": title,
                "authors": get_crossref_authors(paper),
                "year": get_crossref_year(paper),
                "url": url,
            }
        )

    return results


# ============================================================
# 논문 검색 Tool의 실제 함수
# ============================================================

def search_papers(query: str, limit: int = 5) -> str:
    """
    먼저 OpenAlex를 사용하고,
    OpenAlex가 실패하면 Crossref를 fallback으로 사용합니다.
    """

    papers = search_openalex(
        query=query,
        limit=limit,
    )

    used_source = "OpenAlex"

    if papers is None:

        papers = search_crossref(
            query=query,
            limit=limit,
        )

        used_source = "Crossref"

    if not papers:
        return "논문 검색에 실패했거나 검색 결과가 없습니다."

    results = [
        f"[SEARCH SOURCE] {used_source}"
    ]

    for index, paper in enumerate(
        papers,
        start=1,
    ):

        result = (
            f"\n[논문 {index}]\n"
            f"source: {paper['source']}\n"
            f"paper_id: {paper['paper_id']}\n"
            f"title: {paper['title']}\n"
            f"authors: {paper['authors']}\n"
            f"year: {paper['year']}\n"
            f"url: {paper['url']}\n"
        )

        results.append(result)

    return "\n".join(results)


# ============================================================
# OpenAlex 초록 복원
# ============================================================

def reconstruct_abstract(
    inverted_index: dict | None,
) -> str:
    """
    OpenAlex의 abstract_inverted_index를 일반 초록 문자열로 복원합니다.
    """

    if not inverted_index:
        return "초록 정보가 없습니다."

    words = []

    for word, positions in inverted_index.items():

        for position in positions:
            words.append(
                (position, word)
            )

    words.sort(
        key=lambda item: item[0]
    )

    return " ".join(
        word
        for _, word in words
    )


# ============================================================
# OpenAlex 상세 조회
# ============================================================

def get_openalex_details(
    paper_id: str,
) -> str:
    """
    OpenAlex 논문의 상세 정보를 조회합니다.
    """

    openalex_id = (
        paper_id
        .rstrip("/")
        .split("/")[-1]
    )

    url = (
        f"{OPENALEX_SEARCH_URL}/"
        f"{openalex_id}"
    )

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.exceptions.Timeout:
        return (
            "OpenAlex 상세 조회 요청 시간이 "
            "초과되었습니다."
        )

    except requests.exceptions.RequestException as error:
        return (
            "OpenAlex 논문 상세 정보 조회 중 "
            f"오류가 발생했습니다: {error}"
        )

    paper = response.json()

    authorships = paper.get(
        "authorships",
        [],
    )

    authors = ", ".join(
        item.get(
            "author",
            {},
        ).get(
            "display_name",
            "",
        )
        for item in authorships
        if item.get("author")
    )

    abstract = reconstruct_abstract(
        paper.get(
            "abstract_inverted_index"
        )
    )

    return (
        f"source: OpenAlex\n"
        f"title: {paper.get('display_name', '')}\n"
        f"authors: {authors}\n"
        f"year: {paper.get('publication_year', '')}\n"
        f"doi: {paper.get('doi', '')}\n"
        f"citation_count: {paper.get('cited_by_count', 0)}\n"
        f"abstract: {abstract}\n"
    )


# ============================================================
# Crossref 상세 조회
# ============================================================

def get_crossref_details(
    paper_id: str,
) -> str:
    """
    DOI를 이용해 Crossref 논문의 상세 정보를 조회합니다.
    """

    if not paper_id:
        return "Crossref DOI가 없습니다."

    encoded_doi = quote(
        paper_id,
        safe="",
    )

    url = (
        f"{CROSSREF_SEARCH_URL}/"
        f"{encoded_doi}"
    )

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
        )

        response.raise_for_status()

    except requests.exceptions.Timeout:
        return (
            "Crossref 상세 조회 요청 시간이 "
            "초과되었습니다."
        )

    except requests.exceptions.RequestException as error:
        return (
            "Crossref 논문 상세 정보 조회 중 "
            f"오류가 발생했습니다: {error}"
        )

    paper = (
        response
        .json()
        .get("message", {})
    )

    title_list = paper.get(
        "title",
        [],
    )

    title = (
        title_list[0]
        if title_list
        else ""
    )

    abstract = paper.get(
        "abstract",
        "",
    )

    if not abstract:
        abstract = (
            "Crossref에서 제공되는 "
            "초록 정보가 없습니다."
        )

    return (
        f"source: Crossref\n"
        f"title: {title}\n"
        f"authors: {get_crossref_authors(paper)}\n"
        f"year: {get_crossref_year(paper)}\n"
        f"doi: {paper.get('DOI', '')}\n"
        f"url: {paper.get('URL', '')}\n"
        f"abstract: {abstract}\n"
    )


# ============================================================
# 논문 상세 조회 Tool의 실제 함수
# ============================================================

def get_paper_details(
    source: str,
    paper_id: str,
) -> str:
    """
    검색 결과의 source에 따라
    OpenAlex 또는 Crossref에서 상세 정보를 조회합니다.
    """

    source = source.lower().strip()

    if source == "openalex":
        return get_openalex_details(
            paper_id
        )

    if source == "crossref":
        return get_crossref_details(
            paper_id
        )

    return (
        f"지원하지 않는 논문 출처입니다: "
        f"{source}"
    )


# ============================================================
# 보고서 저장 함수
# ============================================================

def save_report(
    content: str,
    filename: str = "paper_report.md",
) -> str:
    """
    논문 탐색 결과를 Markdown 파일로 저장합니다.
    """

    output_dir = Path(__file__).parent

    output_path = (
        output_dir
        / filename
    )

    try:
        output_path.write_text(
            content,
            encoding="utf-8",
        )

    except OSError as error:

        return (
            "보고서 저장 중 오류가 "
            f"발생했습니다: {error}"
        )

    return (
        "보고서가 저장되었습니다: "
        f"{output_path}"
    )


# ============================================================
# LangChain Tool 정의
# ============================================================

@tool("search_papers")
def search_papers_tool(
    query: str,
    limit: int = 5,
) -> str:
    """
    사용자의 연구 주제와 관련된 논문 후보를 검색합니다.

    논문을 처음 탐색할 때 사용하세요.
    """
    return search_papers(
        query=query,
        limit=limit,
    )


@tool("get_paper_details")
def get_paper_details_tool(
    source: str,
    paper_id: str,
) -> str:
    """
    검색된 논문 하나의 상세 정보와 초록을 조회합니다.

    search_papers가 반환한 source와 paper_id를 그대로 사용하세요.
    """
    return get_paper_details(
        source=source,
        paper_id=paper_id,
    )


@tool("save_report")
def save_report_tool(
    content: str,
    filename: str = "paper_report.md",
) -> str:
    """
    최종 논문 조사 결과를 Markdown 파일로 저장합니다.
    """
    return save_report(
        content=content,
        filename=filename,
    )


# ============================================================
# 환경변수와 모델 설정
# ============================================================

load_dotenv()

MODEL = os.getenv(
    "OPENAI_MODEL",
    "gpt-4.1-mini",
)

api_key = os.getenv(
    "OPENAI_API_KEY",
)

if not api_key:
    raise RuntimeError(
        "루트 .env에 OPENAI_API_KEY를 설정하세요."
    )


model = ChatOpenAI(
    model=MODEL,
    timeout=30,
    max_retries=0,
)


TOOLS = [
    search_papers_tool,
    get_paper_details_tool,
    save_report_tool,
]


# ============================================================
# Agent Instructions
# ============================================================

# ============================================================
# Harness 설정
# ============================================================

MAX_RUNS = 3
RECURSION_LIMIT = 30
MIN_DETAIL_PAPERS = 3


# ============================================================
# Agent Instructions
# ============================================================

INSTRUCTIONS = """
당신은 연구자를 돕는 논문 탐색 Agent입니다.

사용자의 연구 질문에 맞는 논문을 Tool을 사용해 실제로 조사하세요.

규칙:

1. 논문 검색이 필요하면 search_papers Tool을 사용하세요.

2. 검색 결과에서 중요한 논문을 선택한 뒤
   get_paper_details Tool을 사용해 초록과 상세 정보를 확인하세요.

3. 검색 결과의 제목만 보고 논문의 내용을 추측하지 마세요.

4. 관련성이 높은 논문을 최소 3편 상세 조회하세요.

5. 첫 검색 결과가 너무 일반적이거나 사용자의 연구 질문과
   직접적인 관련성이 부족하면 검색어를 수정하여
   search_papers를 다시 사용할 수 있습니다.

6. 최종 답변에는 각 논문의
   제목, 저자, 연도, 핵심 내용,
   사용자 연구 주제와의 관련성을 정리하세요.

7. 논문의 관련성을 설명할 때 실제 Tool로 확인한
   제목과 초록의 내용을 근거로 사용하세요.

8. 초록이 직접 뒷받침하지 않는 내용을
   논문의 결론인 것처럼 확대해석하지 마세요.

9. 논문 ID, DOI, URL 등 Tool 결과에 없는 정보를
   만들어내지 마세요.

10. 최종 답변을 작성하기 전에 반드시 save_report Tool을 사용하여
    조사 결과 전체를 paper_report.md 파일로 저장하세요.

11. 보고서를 저장할지 사용자에게 다시 묻지 마세요.
    save_report가 성공한 뒤 최종 답변을 작성하세요.
"""


# ============================================================
# Agent 생성
# ============================================================

agent = create_agent(
    model=model,
    tools=TOOLS,
    system_prompt=INSTRUCTIONS,
)


# ============================================================
# LLM Evaluator
# ============================================================

class PaperEvaluation(TypedDict):
    passed: bool
    feedback: str


EVALUATOR_INSTRUCTIONS = """
당신은 논문 탐색 결과를 검증하는 평가자입니다.

Agent의 답변을 직접 다시 작성하지 마세요.

오직 제공된 사용자 질문, 실제 Tool 검색 결과,
논문 상세 조회 결과와 Agent의 최종 답변만 근거로 평가하세요.

다음을 확인하세요.

1. 사용자의 연구 질문에 실제로 관련 있는 논문을 선택했는가?
2. 최소 3편의 논문을 상세 조회했는가?
3. 각 논문의 관련성 설명이 실제 초록 내용으로 뒷받침되는가?
4. 초록에 없는 leakage 감소 효과나 원인을 과장해서 주장하지 않았는가?
5. 논문들이 사용자의 연구에 도움이 될 정도로 구체적인가?

명확한 문제가 있다면 passed=false로 하고,
다음 실행에서 바로 활용할 수 있는 구체적인 feedback을 작성하세요.

문제가 없다면 passed=true와 빈 feedback을 반환하세요.
"""


evaluator = model.with_structured_output(
    PaperEvaluation,
    strict=True,
)


# ============================================================
# Trace 분석
# ============================================================

def get_successful_tool_results(
    messages,
    tool_name: str | None = None,
) -> list[ToolMessage]:
    """
    성공적으로 실행된 ToolMessage만 가져옵니다.
    """

    results = []

    for message in messages:

        if not isinstance(message, ToolMessage):
            continue

        if getattr(message, "status", "success") == "error":
            continue

        if tool_name is not None and message.name != tool_name:
            continue

        results.append(message)

    return results


def get_final_answer(messages) -> str:
    """
    Tool Call이 없는 마지막 AIMessage를 최종 답변으로 가져옵니다.
    """

    for message in reversed(messages):

        if (
            isinstance(message, AIMessage)
            and not message.tool_calls
        ):
            return str(message.content)

    return ""


def print_agent_trace(messages) -> None:
    """
    Agent의 Tool Call과 Tool Result를 실행 순서대로 출력합니다.
    """

    print()
    print("========== AGENT TRACE ==========")

    for message in messages:

        if (
            isinstance(message, AIMessage)
            and message.tool_calls
        ):

            for call in message.tool_calls:

                print()
                print(
                    "[TOOL CALL]",
                    call["name"],
                    call["args"],
                )

        elif isinstance(message, ToolMessage):

            print()
            print(
                "[TOOL RESULT]",
                message.name,
            )

            print(
                str(message.content)[:1500]
            )


# ============================================================
# 규칙 기반 Verification
# ============================================================

def verify_tool_usage(messages) -> list[str]:
    """
    필요한 Tool이 실제로 사용되었는지 검사합니다.
    """

    issues = []

    search_results = get_successful_tool_results(
        messages,
        "search_papers",
    )

    detail_results = get_successful_tool_results(
        messages,
        "get_paper_details",
    )

    save_results = get_successful_tool_results(
        messages,
        "save_report",
    )

    if not search_results:
        issues.append(
            "search_papers Tool을 사용해 논문을 검색하세요."
        )

    if len(detail_results) < MIN_DETAIL_PAPERS:
        issues.append(
            f"get_paper_details로 최소 "
            f"{MIN_DETAIL_PAPERS}편의 논문을 상세 조회하세요. "
            f"현재 상세 조회 수: {len(detail_results)}"
        )

    if not save_results:
        issues.append(
            "최종 보고서를 save_report Tool로 저장하세요."
        )

    return issues


# ============================================================
# LLM Verification
# ============================================================

def evaluate_with_llm(
    task: str,
    messages,
    answer: str,
) -> tuple[bool, str]:
    """
    실제 상세 조회 결과를 근거로
    Agent 답변의 관련성과 사실성을 평가합니다.
    """

    detail_results = get_successful_tool_results(
        messages,
        "get_paper_details",
    )

    evidence = "\n\n".join(
        f"[상세 조회 논문 {index}]\n{message.content}"
        for index, message in enumerate(
            detail_results,
            start=1,
        )
    )

    evaluation_input = f"""
사용자의 연구 질문:

{task}


실제 get_paper_details Tool 결과:

{evidence or "(상세 조회 결과 없음)"}


Agent의 최종 답변:

{answer}
"""

    result = evaluator.invoke(
        [
            {
                "role": "system",
                "content": EVALUATOR_INSTRUCTIONS,
            },
            {
                "role": "user",
                "content": evaluation_input,
            },
        ]
    )

    return (
        result["passed"],
        result["feedback"].strip(),
    )


# ============================================================
# 종합 Verification
# ============================================================

def verify_run(
    task: str,
    messages,
) -> tuple[bool, str, list[str]]:
    """
    규칙 검증과 LLM Evaluator를 결합합니다.
    """

    answer = get_final_answer(messages)

    issues = verify_tool_usage(messages)

    if not answer:
        issues.append(
            "Agent의 최종 답변이 없습니다."
        )

    llm_passed = False
    llm_feedback = ""

    if answer:

        try:
            llm_passed, llm_feedback = evaluate_with_llm(
                task,
                messages,
                answer,
            )

        except Exception as error:

            issues.append(
                f"LLM Evaluator 실행 오류: "
                f"{type(error).__name__}"
            )

    if not llm_passed and llm_feedback:

        issues.append(
            "LLM 평가: " + llm_feedback
        )

    passed = (
        not issues
        and llm_passed
    )

    if passed:

        feedback = ""

    else:

        feedback = (
            "이전 실행은 검증을 통과하지 못했습니다.\n"
            "다음 문제를 수정하여 논문 탐색을 다시 수행하세요:\n- "
            + "\n- ".join(issues)
        )

    return (
        passed,
        feedback,
        issues,
    )


# ============================================================
# 한 번의 Agent Run
# ============================================================

def run_agent_once(
    task: str,
    feedback: str = "",
):
    """
    Agent를 한 번 실행합니다.

    feedback이 있으면 이전 실행의 문제를
    사용자 요청과 함께 전달합니다.
    """

    if feedback:

        run_task = f"""
원래 사용자 요청:

{task}

이전 실행에 대한 검증 Feedback:

{feedback}

Feedback을 반영하여 처음부터 다시 논문을 조사하세요.
이전 답변을 그대로 반복하지 말고 필요한 경우 검색어와
선택 논문을 변경하세요.
"""

    else:

        run_task = task

    try:

        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=run_task
                    )
                ]
            },
            config={
                "recursion_limit": RECURSION_LIMIT
            },
        )

        return result

    except GraphRecursionError:

        print()
        print(
            "[HARNESS STOP] "
            "Agent가 최대 실행 단계에 도달했습니다."
        )

        return None

    except Exception as error:

        print()
        print(
            "[HARNESS ERROR]",
            type(error).__name__,
            str(error),
        )

        return None


# ============================================================
# Outer Verification Loop
# ============================================================

def run_verified_agent(
    task: str,
    max_runs: int = MAX_RUNS,
):
    """
    Agent 실행 → Verification → Feedback → Re-run을 반복합니다.
    """

    feedback = ""

    for run_number in range(
        1,
        max_runs + 1,
    ):

        print()
        print(
            "=" * 60
        )

        print(
            f"[RUN {run_number}/{max_runs}]"
        )

        print(
            "=" * 60
        )

        result = run_agent_once(
            task=task,
            feedback=feedback,
        )

        if result is None:

            print(
                "[RUN STATUS] ERROR"
            )

            return None

        messages = result["messages"]

        print_agent_trace(
            messages
        )

        answer = get_final_answer(
            messages
        )

        passed, feedback, issues = verify_run(
            task=task,
            messages=messages,
        )

        print()
        print(
            "========== VERIFICATION =========="
        )

        if passed:

            print(
                "[VERIFY] PASS"
            )

            print()
            print(
                "========== FINAL ANSWER =========="
            )

            print(
                answer
            )

            print()
            print(
                f"[RUN STATUS] COMPLETED "
                f"(run={run_number})"
            )

            return result

        print(
            "[VERIFY] FAIL"
        )

        for issue in issues:

            print(
                "-",
                issue,
            )

        # 최대 Run에 도달한 경우 더 이상 재실행하지 않음
        if run_number >= max_runs:

            print()
            print(
                "[LOOP STOP] "
                "MAX_RUNS에 도달했습니다."
            )

            print()
            print(
                "========== LAST ANSWER =========="
            )

            print(
                answer
            )

            return result

        print()
        print(
            "[FEEDBACK]"
        )

        print(
            feedback
        )

        print()
        print(
            "[RE-RUN] Feedback을 반영해 "
            "새 Agent Run을 시작합니다."
        )

    return None


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":

    task = input("연구 주제를 입력하세요: ").strip()

    if not task:
        print("연구 주제가 입력되지 않았습니다.")

    else:
        print()
        print("[USER TASK]")
        print(task)

        run_verified_agent(task)