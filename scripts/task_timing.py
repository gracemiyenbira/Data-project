import os, sqlite3
from datetime import datetime
from collections import defaultdict

AIRFLOW_HOME = os.environ.get("AIRFLOW_HOME", os.path.join(os.path.expanduser("~"), "airflow"))
DB = os.path.join(AIRFLOW_HOME, "airflow.db")
DAG = "group3_surveillance_pipeline"
# Runs that completed successfully while running alone (the two runs that overlapped
# at 21:31 UTC on 29 Sep are excluded because they shared CPU)
SOLO = ["manual__2026-09-27T18:52", "manual__2026-09-27T19:17", "manual__2026-09-27T19:43",
        "manual__2026-09-28T18:02", "manual__2026-09-29T21:33", "manual__2026-09-29T21:41"]

def secs(a, b):
    return (datetime.fromisoformat(str(b)) - datetime.fromisoformat(str(a))).total_seconds()

con = sqlite3.connect(DB)
runs = [r for (r,) in con.execute(
    "SELECT run_id FROM dag_run WHERE dag_id=? AND state='success'", (DAG,))
    if any(r.startswith(p) for p in SOLO)]
print(f"{len(runs)} solo successful runs found\n")

per_task, run_total = defaultdict(list), {}
for run in runs:
    rows = con.execute(
        "SELECT task_id, start_date, end_date FROM task_instance "
        "WHERE dag_id=? AND run_id=? AND state='success'", (DAG, run)).fetchall()
    for t, s, e in rows:
        if s and e:
            per_task[t].append(secs(s, e))
    starts = [r[1] for r in rows if r[1]]; ends = [r[2] for r in rows if r[2]]
    if starts and ends:
        run_total[run] = secs(min(starts), max(ends))

print(f"{'task':28s}{'n':>3}{'min s':>9}{'mean s':>9}{'max s':>9}{'% of mean run':>15}")
if not run_total:
    raise RuntimeError("No selected successful runs with task timestamps were found in the Airflow metadata database.")

mean_run = sum(run_total.values()) / len(run_total)
for t, v in sorted(per_task.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
    m = sum(v) / len(v)
    print(f"{t:28s}{len(v):>3}{min(v):>9.2f}{m:>9.2f}{max(v):>9.2f}{100 * m / mean_run:>14.1f}%")
print("\nWhole-run time (first task start to last task end):")
for r, s in sorted(run_total.items()):
    print(f"  {r[:29]}  {s:6.2f} s")
print(f"  mean {mean_run:.2f} s")
