#!/bin/bash

# Script to index a directory using the Python CLI

DEFAULT_MODEL_NAME="all-MiniLM-L6-v2" # Keep in sync with cli.py or make it an arg here too
MODEL_NAME="$DEFAULT_MODEL_NAME"
FORCE_INDEX=""
DIRECTORY_TO_INDEX=""

# Parse arguments for this shell script
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL_NAME="$2"; shift ;;
        --force) FORCE_INDEX="--force" ;;
        -h|--help)
            echo "Usage: $0 <directory_to_index> [--model <model_name>] [--force]"
            echo "Options:"
            echo "  <directory_to_index>  Path to the directory to index (required)."
            echo "  --model <model_name>  Sentence Transformer model to use (default: $DEFAULT_MODEL_NAME)."
            echo "  --force               Force re-indexing even if an index exists."
            echo "  -h, --help            Show this help message."
            exit 0
            ;;
        *) DIRECTORY_TO_INDEX="$1" ;;
    esac
    shift
done

if [ -z "$DIRECTORY_TO_INDEX" ]; then
  echo "Error: Directory to index is required."
  echo "Usage: $0 <directory_to_index> [--model <model_name>] [--force]"
  exit 1
fi

if [ ! -d "$DIRECTORY_TO_INDEX" ]; then
    echo "Error: '$DIRECTORY_TO_INDEX' is not a valid directory."
    exit 1
fi

# Assuming the script is run from the project root, or adjust path to cli.py
PYTHON_CLI_PATH="src/cli.py" 

if [ ! -f "$PYTHON_CLI_PATH" ]; then
    echo "Error: Python CLI script not found at $PYTHON_CLI_PATH. Make sure you are in the project root."
    exit 1
fi

echo "Starting indexing..."
echo "Directory: $DIRECTORY_TO_INDEX"
echo "Model: $MODEL_NAME"
if [ -n "$FORCE_INDEX" ]; then
    echo "Force re-indexing: Yes"
fi

# Construct the command
COMMAND="python $PYTHON_CLI_PATH --model "$MODEL_NAME" index "$DIRECTORY_TO_INDEX" $FORCE_INDEX"
echo "Executing: $COMMAND"
eval $COMMAND

echo "Indexing script finished."
