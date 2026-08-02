from typing import TypedDict, List, Dict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
import requests
import operator
import json
import uuid


llm = ChatOllama(model="qwen2.5:7b")
llm_deterministic = ChatOllama(model="qwen2.5:7b", temperature=0)  # for categorization


class TriageState(TypedDict):
    repo:            str                          # "owner/repo"
    issues:          List[Dict]                    # raw fetched issues
    triaged_issues:  Annotated[List[Dict], operator.add]  # results, parallel-safe
    final_report:    str


def fetch_issues(owner: str, repo: str, limit: int = 10) -> list:
    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    try:
        # fetch more than needed since PRs will be filtered out
        response = requests.get(url, params={"state": "open", "per_page": limit * 10}, timeout=10)
        if response.status_code == 404:
            return []
        if response.status_code == 403:
            print("  Rate limited by GitHub API")
            return []
        issues = response.json()
        issues = [i for i in issues if "pull_request" not in i]
        return issues[:limit]  # trim back to the requested amount
    except Exception as e:
        print(f"  Failed to fetch issues: {e}")
        return []

def fetch_node(state: TriageState) -> dict:
    repo = state["repo"]
    
    if "/" not in repo:
        print(f"\n  Invalid repo format: '{repo}' — expected 'owner/repo'")
        return {"issues": [], "triaged_issues": []}
    
    owner, repo_name = repo.split("/")
    issues = fetch_issues(owner, repo_name, limit=10)
    print(f"\n  Fetched {len(issues)} issues from {state['repo']}")
    return {"issues": issues, "triaged_issues": []}


def route_to_triage(state: TriageState):
    issues = state["issues"]
    if not issues:
        return "synthesize"  # nothing to triage, skip straight to report
    return [Send("triage_one", {"issue": issue, "all_issues": issues}) for issue in issues]

def triage_one_node(state: dict) -> dict:
    issue = state["issue"]
    title = issue.get("title", "")
    body = issue.get("body", "") or "(no description provided)"
    number = issue.get("number")

    print(f"\n  [Triage] Issue #{number}: {title}")

    try:
        category_response = llm_deterministic.invoke(
            f"Categorize this GitHub issue as exactly one of: Bug, Feature Request, "
            f"Documentation, Question, Other. Return ONLY the category word.\n\n"
            f"Title: {title}\nBody: {body[:500]}"
        )
        category = category_response.content.strip()

        response_draft = llm.invoke(
            f"Draft a brief, helpful triage response (2-3 sentences) for this GitHub issue. "
            f"Be professional and specific.\n\nTitle: {title}\nBody: {body[:500]}"
        )
        draft = response_draft.content.strip()

    except Exception as e:
        print(f"  Triage failed for #{number}: {e}")
        category = "Unknown"
        draft = "Could not generate a response due to an error."

    return {
        "triaged_issues": [{
            "number": number,
            "title": title,
            "category": category,
            "draft_response": draft
        }]
    }

def synthesize_node(state: TriageState) -> dict:
    repo = state["repo"]
    triaged = state["triaged_issues"]

    if not triaged:
        return {"final_report": f"No open issues found for {repo}, or the repo could not be accessed."}

    report = f"GITHUB TRIAGE REPORT — {repo}\n"
    report += f"{'=' * 50}\n\n"

    categories = {}
    for item in triaged:
        cat = item["category"]
        categories[cat] = categories.get(cat, 0) + 1

    report += "Summary: " + ", ".join(f"{count} {cat}" for cat, count in categories.items()) + "\n\n"

    for item in triaged:
        report += f"Issue #{item['number']}: {item['title']}\n"
        report += f"  Category: {item['category']}\n"
        report += f"  Draft response: {item['draft_response']}\n\n"

    return {"final_report": report}


graph = StateGraph(TriageState)
graph.add_node("fetch", fetch_node)
graph.add_node("triage_one", triage_one_node)
graph.add_node("synthesize", synthesize_node)

graph.set_entry_point("fetch")
graph.add_conditional_edges("fetch", route_to_triage, ["triage_one", "synthesize"])
graph.add_edge("triage_one", "synthesize")
graph.add_edge("synthesize", END)

app = graph.compile()


if __name__ == "__main__":
    with SqliteSaver.from_conn_string("triage_checkpoints.db") as checkpointer:
        app = graph.compile(checkpointer=checkpointer)

        repo = input("Enter a GitHub repo (owner/repo): ")
        # config = {"configurable": {"thread_id": f"triage-{repo.replace('/', '-')}"}}
        config = {"configurable": {"thread_id": f"triage-{repo.replace('/', '-')}-{uuid.uuid4().hex[:8]}"}}
        result = app.invoke({
            "repo": repo,
            "issues": [],
            "triaged_issues": [],
            "final_report": ""
        }, config=config)

        print("\n" + "=" * 50)
        print(result["final_report"])