#!/bin/bash

# run_eval.sh - GPU Queue System for Model Evaluation
# This script runs evaluate_with_steering.py on multiple GPUs in parallel
# using a queue system to ensure efficient GPU utilization.

# Default values
GPUS="0,1,2,3,4,5,6,7"
MODEL_SIZE="7b"
MULTIPLIERS="-2,-1,0,1,2"
TASKS="gsm8k"
NUM_FEWSHOT=0
USE_BASE_MODEL=false
OVERWRITE=false
LIMIT=""
LAYERS="9,10,11"
BEHAVIORS="reasoning"
SYSTEM_PROMPT="pos"
LOG_DIR="eval_logs"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --model_size)
      MODEL_SIZE="$2"
      shift 2
      ;;
    --multipliers)
      MULTIPLIERS="$2"
      shift 2
      ;;
    --tasks)
      TASKS="$2"
      shift 2
      ;;
    --num_fewshot)
      NUM_FEWSHOT="$2"
      shift 2
      ;;
    --use_base_model)
      USE_BASE_MODEL=true
      shift
      ;;
    --overwrite)
      OVERWRITE=true
      shift
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --layers)
      LAYERS="$2"
      shift 2
      ;;
    --behaviors)
      BEHAVIORS="$2"
      shift 2
      ;;
    --system_prompt)
      SYSTEM_PROMPT="$2"
      shift 2
      ;;
    --log_dir)
      LOG_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Convert comma-separated strings to arrays
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
IFS=',' read -ra LAYER_ARRAY <<< "$LAYERS"
IFS=',' read -ra MULTIPLIER_ARRAY <<< "$MULTIPLIERS"
IFS=',' read -ra TASK_ARRAY <<< "$TASKS"
IFS=',' read -ra BEHAVIOR_ARRAY <<< "$BEHAVIORS"

# Print configuration
echo "=== Evaluation Configuration ==="
echo "GPUs: $GPUS"
echo "Model Size: $MODEL_SIZE"
echo "Layers: $LAYERS"
echo "Multipliers: $MULTIPLIERS"
echo "Tasks: $TASKS"
echo "Behaviors: $BEHAVIORS"
echo "Num Few-shot: $NUM_FEWSHOT"
echo "System Prompt: $SYSTEM_PROMPT"
echo "Use Base Model: $USE_BASE_MODEL"
echo "Overwrite: $OVERWRITE"
if [ ! -z "$LIMIT" ]; then
  echo "Limit: $LIMIT"
fi
echo "=============================="

# Build the command queue
declare -a COMMAND_QUEUE

for behavior in "${BEHAVIOR_ARRAY[@]}"; do
  for task in "${TASK_ARRAY[@]}"; do
    for layer in "${LAYER_ARRAY[@]}"; do
      for multiplier in "${MULTIPLIER_ARRAY[@]}"; do
        # Create the base command
        cmd="python evaluate_with_steering.py \
          --model_size $MODEL_SIZE \
          --layers $layer \
          --multipliers $multiplier \
          --behaviors $behavior \
          --tasks $task \
          --num_fewshot $NUM_FEWSHOT \
          --system_prompt $SYSTEM_PROMPT"

        # Add optional flags
        if [ "$USE_BASE_MODEL" = true ]; then
          cmd="$cmd --use_base_model"
        fi

        if [ "$OVERWRITE" = true ]; then
          cmd="$cmd --overwrite"
        fi

        if [ ! -z "$LIMIT" ]; then
          cmd="$cmd --limit $LIMIT"
        fi

        # Add to queue
        COMMAND_QUEUE+=("$cmd")
      done
    done
  done
done

echo "Generated ${#COMMAND_QUEUE[@]} evaluation commands"

# Function to run a command on a specific GPU
run_on_gpu() {
  local gpu=$1
  local cmd=$2
  local timestamp=$(date +"%Y%m%d_%H%M%S")
  local log_file="$LOG_DIR/gpu${gpu}_${timestamp}.log"

  echo "[$(date)] Starting job on GPU $gpu"
  echo "Command: $cmd"
  echo "Log: $log_file"

  # Set environment variable for GPU and run the command
  export CUDA_VISIBLE_DEVICES=$gpu
  export PYTHONPATH="/data_x/junkim100/lm-evaluation-harness:$PYTHONPATH"
  eval $cmd > "$log_file" 2>&1

  local status=$?
  echo "[$(date)] Job on GPU $gpu completed with status $status"
  return $status
}

# Function to check if a GPU is available
is_gpu_available() {
  local gpu=$1
  local processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits --id=$gpu | wc -l)

  if [ "$processes" -eq 0 ]; then
    return 0  # GPU is available
  else
    return 1  # GPU is busy
  fi
}

# Function to get the next available GPU
get_next_available_gpu() {
  while true; do
    for gpu in "${GPU_ARRAY[@]}"; do
      if is_gpu_available $gpu; then
        echo $gpu
        return 0
      fi
    done

    # No GPU available, wait and try again
    echo "Waiting for an available GPU..."
    sleep 10
  done
}

# Process queue with GPU parallelism
echo "Starting evaluation with ${#GPU_ARRAY[@]} GPUs"
echo "Queue contains ${#COMMAND_QUEUE[@]} jobs"

# Track running processes
declare -A RUNNING_JOBS

# Start initial jobs on each GPU
for gpu in "${GPU_ARRAY[@]}"; do
  if [ ${#COMMAND_QUEUE[@]} -gt 0 ]; then
    # Get next command
    cmd="${COMMAND_QUEUE[0]}"
    # Remove it from the queue
    COMMAND_QUEUE=("${COMMAND_QUEUE[@]:1}")

    # Run in background
    run_on_gpu $gpu "$cmd" &
    pid=$!
    RUNNING_JOBS[$pid]=$gpu

    echo "Started job on GPU $gpu (PID: $pid, ${#COMMAND_QUEUE[@]} jobs remaining)"
  fi
done

# Process remaining commands as GPUs become available
while [ ${#COMMAND_QUEUE[@]} -gt 0 ]; do
  # Wait for any child process to exit
  wait -n
  exit_code=$?

  # Find which process completed
  for pid in "${!RUNNING_JOBS[@]}"; do
    if ! kill -0 $pid 2>/dev/null; then
      # This process has completed
      gpu=${RUNNING_JOBS[$pid]}
      unset RUNNING_JOBS[$pid]

      echo "Job on GPU $gpu completed (PID: $pid, exit code: $exit_code)"

      # Start a new job on this GPU
      if [ ${#COMMAND_QUEUE[@]} -gt 0 ]; then
        # Get next command
        cmd="${COMMAND_QUEUE[0]}"
        # Remove it from the queue
        COMMAND_QUEUE=("${COMMAND_QUEUE[@]:1}")

        # Run in background
        run_on_gpu $gpu "$cmd" &
        pid=$!
        RUNNING_JOBS[$pid]=$gpu

        echo "Started job on GPU $gpu (PID: $pid, ${#COMMAND_QUEUE[@]} jobs remaining)"
      fi

      break
    fi
  done
done

# Wait for all remaining jobs to complete
echo "All jobs have been assigned. Waiting for remaining ${#RUNNING_JOBS[@]} jobs to complete..."
wait

echo "All evaluations completed successfully!"
echo "Logs are available in the $LOG_DIR directory"
