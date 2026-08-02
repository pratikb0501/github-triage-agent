import json
from agent import fetch_issues, triage_one_node

def load_test_set(filepath="test_set.json"):
    with open(filepath, "r") as f:
        return json.load(f)


def run_evaluation(test_set):
    results = []

    for expected in test_set:
        # build a minimal issue dict to pass into triage_one_node
        issue = {
            "number": expected["number"],
            "title": expected["title"],
            "body": expected.get("body", "")
        }

        # run the SAME triage logic your agent uses
        output = triage_one_node({"issue": issue, "all_issues": []})
        actual_category = output["triaged_issues"][0]["category"]

        is_match = actual_category.strip().lower() == expected["correct_category"].strip().lower()

        results.append({
            "number": expected["number"],
            "title": expected["title"],
            "expected": expected["correct_category"],
            "actual": actual_category,
            "match": is_match
        })

    return results


def print_report(results):
    correct = sum(1 for r in results if r["match"])
    total = len(results)
    accuracy = (correct / total) * 100 if total > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"EVALUATION REPORT")
    print(f"{'=' * 60}")
    print(f"Accuracy: {correct}/{total} ({accuracy:.1f}%)\n")

    for r in results:
        status = "✅" if r["match"] else "❌"
        print(f"{status} #{r['number']}: {r['title'][:50]}")
        print(f"   Expected: {r['expected']}  |  Actual: {r['actual']}")

    print(f"\n{'=' * 60}")
    print(f"FINAL SCORE: {accuracy:.1f}%")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    test_set = load_test_set()
    results = run_evaluation(test_set)
    print_report(results)