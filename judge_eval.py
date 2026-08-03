import json
from agent import llm_deterministic  # reuse temperature=0 client
from agent import triage_one_node


def judge_response(issue_title: str, issue_body: str, draft_response: str) -> dict:
    """Score a draft response using the LLM as judge. Same model, temp=0,
    with the known self-preference bias limitation acknowledged."""

    prompt = f"""You are evaluating a draft response to a GitHub issue.
Rate the response on a scale of 1-5 for each criterion below.

1. Professional: is the tone professional and courteous?
2. Relevant: does it specifically address this issue (not generic)?
3. Concise: is it to the point, without unnecessary padding?

Issue title: {issue_title}
Issue body: {issue_body[:300]}
Draft response: {draft_response}

Return ONLY valid JSON in this exact format, nothing else:
{{"professional": <1-5>, "relevant": <1-5>, "concise": <1-5>}}
"""

    result = llm_deterministic.invoke(prompt)
    raw = result.content.strip().replace("```json", "").replace("```", "").strip()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        scores = {"professional": None, "relevant": None, "concise": None}

    return scores


def sanity_check_judge():
    """One-time validation: confirm the judge actually discriminates between
    a deliberately bad response and a good one, rather than defaulting to a
    fixed score regardless of quality. Run this whenever the judge prompt
    changes, to re-confirm it still discriminates correctly."""

    bad = judge_response(
        "App crashes on login",
        "",
        "lol idk sounds like a you problem, not my issue",
    )
    good = judge_response(
        "App crashes on login",
        "",
        "Thank you for reporting this. Could you share your OS version and steps to reproduce?",
    )

    print("Judge sanity check")
    print(f"  Bad response scores:  {bad}")
    print(f"  Good response scores: {good}")

    if bad["professional"] is None or good["professional"] is None:
        print("  ⚠️  Could not parse judge output — check the judge prompt/model")
    elif bad["professional"] >= good["professional"]:
        print("  ⚠️  WARNING: judge did not discriminate — scores may not be trustworthy")
    else:
        print("  ✅ Judge correctly scored the bad response lower — discriminates properly")


def run_judge_eval(test_set_path="test_set.json"):
    with open(test_set_path, "r") as f:
        test_set = json.load(f)

    print("\n" + "=" * 60)
    print("LLM-AS-JUDGE EVALUATION")
    print("(Note: same model judging its own output — see README for caveat)")
    print("=" * 60)

    for item in test_set:
        issue = {"number": item["number"], "title": item["title"], "body": ""}

        # generate the draft response using the real triage logic
        output = triage_one_node({"issue": issue, "all_issues": []})
        draft = output["triaged_issues"][0]["draft_response"]

        # judge it
        scores = judge_response(item["title"], "", draft)

        print(f"\n#{item['number']}: {item['title'][:50]}")
        print(f"  Professional: {scores['professional']}/5")
        print(f"  Relevant:     {scores['relevant']}/5")
        print(f"  Concise:      {scores['concise']}/5")


if __name__ == "__main__":
    sanity_check_judge()   # runs once, at the top
    run_judge_eval()       # then scores the real test set