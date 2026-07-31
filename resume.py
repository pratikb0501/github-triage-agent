from agent import app, TriageState
from langgraph.checkpoint.sqlite import SqliteSaver

repo = input("Which repo were you resuming? (owner/repo): ")

with SqliteSaver.from_conn_string("triage_checkpoints.db") as checkpointer:
    from agent import graph
    resumed_app = graph.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": f"triage-{repo.replace('/', '-')}"}}

    state = resumed_app.get_state(config)
    print("Issues fetched so far:", len(state.values.get("issues", [])))
    print("Issues triaged so far:", len(state.values.get("triaged_issues", [])))

    result = resumed_app.invoke(None, config=config)
    print("\n" + result["final_report"])