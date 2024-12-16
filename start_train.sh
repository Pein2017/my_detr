#!/bin/bash

# Default values
ROOT_DIR="/data/training_code/Pein/DETR/my_detr"
EXPERIMENT="debug"

# Function to display usage information
usage() {
    echo "Usage: $0 [-e|--experiment <experiment_name>]"
    echo
    echo "Options:"
    echo "  -e, --experiment    Specify the experiment name (default: debug)"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -e|--experiment)
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                EXPERIMENT="$2"
                shift 2
            else
                echo "Error: --experiment requires a non-empty argument."
                usage
            fi
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Function to handle interrupts and cleanup
cleanup() {
    echo -e "\nReceived interrupt signal. Cleaning up..."
    # Kill all child processes in the same process group
    pkill -P $$
    # Kill any remaining Python processes
    pkill -f "torchrun"
    pkill -f "python launch_train.py"
    exit 1
}

# Register the cleanup function for multiple signals
trap cleanup SIGINT SIGTERM SIGHUP

# Function to format seconds into days:hours:minutes
format_time() {
    local seconds=$1
    local days=$((seconds/86400))
    local hours=$(( (seconds%86400)/3600 ))
    local minutes=$(( (seconds%3600)/60 ))
    
    if [ $days -gt 0 ]; then
        echo "${days}d:${hours}h:${minutes}m"
    elif [ $hours -gt 0 ]; then
        echo "${hours}h:${minutes}m"
    else
        echo "${minutes}m"
    fi
}

# Function to get number of GPUs from experiment config
get_num_gpus() {
    local exp_name=$1
    local config_path="${ROOT_DIR}/configs/exp/${exp_name}.yaml"
    
    if [ -f "$config_path" ]; then
        # Use awk to find num_gpus under distributed section, regardless of position
        local num_gpus=$(awk '/^distributed:/{p=1} p&&/num_gpus:/{print $2;exit}' "$config_path")
        if [ -n "$num_gpus" ]; then
            echo "$num_gpus"
            return
        fi
    fi
    
    # Default to 1 if not found
    echo "1"
}

# Get number of GPUs based on experiment
NUM_GPUS=$(get_num_gpus "$EXPERIMENT")

# Change to root directory
cd "${ROOT_DIR}" || { echo "Failed to change directory to ${ROOT_DIR}"; exit 1; }

# Calculate optimal OMP_NUM_THREADS based on CPU cores and GPU count
CPU_CORES=$(nproc)
THREADS_PER_GPU=$((CPU_CORES / NUM_GPUS))
# Ensure at least 1 thread per GPU
if [ $THREADS_PER_GPU -lt 1 ]; then
    THREADS_PER_GPU=1
fi
export OMP_NUM_THREADS=$THREADS_PER_GPU

echo "Starting training at: $(date)"
echo "Using ${NUM_GPUS} GPUs with experiment: ${EXPERIMENT}"
echo "CPU cores: ${CPU_CORES}, Threads per GPU: ${THREADS_PER_GPU}"
start_time=$(date +%s)

# Run training with error handling
if [ -n "$EXPERIMENT" ]; then
    echo "Starting training with experiment: $EXPERIMENT"
    # Run with torchrun for DDP
    (
        torchrun \
            --nproc_per_node="$NUM_GPUS" \
            --master_port=$(shuf -i 29500-29999 -n 1) \
            launch_train.py --config-name default +exp="$EXPERIMENT"
    ) &
    
    # Store the PID of the subshell
    training_pid=$!
    
    # Wait for the process to complete
    wait "$training_pid"
    exit_status=$?
    
    if [ "$exit_status" -ne 0 ]; then
        echo "Training failed with exit status: $exit_status"
        cleanup
    fi
else
    echo "Starting training with default config"
    (
        torchrun \
            --nproc_per_node=1 \
            --master_port=$(shuf -i 29500-29999 -n 1) \
            launch_train.py --config-name default
    ) &
    
    training_pid=$!
    wait "$training_pid"
    exit_status=$?
    
    if [ "$exit_status" -ne 0 ]; then
        echo "Training failed with exit status: $exit_status"
        cleanup
    fi
fi

# Calculate and print total time
end_time=$(date +%s)
total_seconds=$((end_time - start_time))
formatted_time=$(format_time "$total_seconds")

echo "Training completed at: $(date)"
echo "Total training time: ${formatted_time}"

# Ensure all child processes are cleaned up
cleanup