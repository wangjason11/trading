#!/usr/bin/env bash
# PLAN_J: the session-4 random-tail corpus through one tree (PYTHONPATH = the tree), JOBS seeds at a time.
#   bash search_tree.sh TREE OUTDIR [JOBS]
# WAVES-B seeds 401-424 x 2500, WAVES 301-312 x 600, REAL 201-212 x 400 (PLAN_I §9.10 session 4; R_k = 200+k etc.).
TREE="$1"; OUT="$2"; JOBS="${3:-24}"
S="$TREE/engine_v2/plans/plan_e_inputs/review_scripts/random_tail_search.py"
mkdir -p "$OUT"
run() { PYTHONPATH="$TREE" python "$S" "$1" "$OUT/$2_$3.jsonl" "$3" "$4" > "$OUT/$2_$3.log" 2>&1; }
export -f run; export TREE OUT S
{ for s in $(seq 401 424); do echo "waves WB $s 2500"; done
  for s in $(seq 301 312); do echo "waves W $s 600"; done
  for s in $(seq 201 212); do echo "real R $s 400"; done; } | xargs -P "$JOBS" -L 1 bash -c 'run "$0" "$1" "$2" "$3"'
