#!/bin/bash
export PYTHONUNBUFFERED=1
export MLFLOW_ALLOW_FILE_STORE=true
source /home/jo/fish_env/bin/activate

run_if_needed() {
    local script=$1
    local log=$2
    local done_marker=$3

    if [ -f "$done_marker" ]; then
        echo "$(basename $script) already complete — skipping"
    else
        echo "Running $(basename $script)..."
        python3 $script > $log 2>&1
        if [ $? -eq 0 ]; then touch $done_marker; fi
        echo "$(basename $script) done"
    fi
}

RUNS="/mnt/LinuxBackup/jo/runs"
SCRIPTS="/mnt/LinuxBackup/jo/scripts/training"
LOGS="/mnt/LinuxBackup/jo/logs/training"

run_if_needed $SCRIPTS/train_v8s_human.py   $LOGS/v8s_human.log   $RUNS/v8s_human.done
run_if_needed $SCRIPTS/train_v8s_gdtiny.py  $LOGS/v8s_gdtiny.log  $RUNS/v8s_gdtiny.done
run_if_needed $SCRIPTS/train_v8s_gdbase.py  $LOGS/v8s_gdbase.log  $RUNS/v8s_gdbase.done
run_if_needed $SCRIPTS/train_v11x_human.py  $LOGS/v11x_human.log  $RUNS/v11x_human.done
run_if_needed $SCRIPTS/train_v11x_gdtiny.py $LOGS/v11x_gdtiny.log $RUNS/v11x_gdtiny.done
run_if_needed $SCRIPTS/train_v11x_gdbase.py $LOGS/v11x_gdbase.log $RUNS/v11x_gdbase.done
run_if_needed $SCRIPTS/train_rfdetr_human.py   $LOGS/rfdetr_human.log   $RUNS/rfdetr_human.done
run_if_needed $SCRIPTS/train_rfdetr_gdtiny.py  $LOGS/rfdetr_gdtiny.log  $RUNS/rfdetr_gdtiny.done
run_if_needed $SCRIPTS/train_rfdetr_gdbase.py  $LOGS/rfdetr_gdbase.log  $RUNS/rfdetr_gdbase.done

echo "All training complete"
