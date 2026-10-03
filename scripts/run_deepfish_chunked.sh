#!/bin/bash
export PYTHONUNBUFFERED=1 MLFLOW_ALLOW_FILE_STORE=true
for m in human gdtiny gdbase; do
  for s in 0 500 1000 1500 2000 2500; do
    python rfdetr_deepfish_chunked.py infer $m $s $((s+500)) || exit 1
  done
  python rfdetr_deepfish_chunked.py metrics $m || exit 1
done
echo "ALL DONE"
