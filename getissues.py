import sys
sys.path.append('.')  # so we can import from agent.py
from agent import fetch_issues
import json

issues = fetch_issues("psf", "requests", limit=10)

print(f"Fetched {len(issues)} issues\n")
for i, issue in enumerate(issues, 1):
    print(f"{i}. #{issue['number']}: {issue['title']}")
    body = issue.get('body') or "(no description)"
    print(f"   {body[:150]}...")
    print()