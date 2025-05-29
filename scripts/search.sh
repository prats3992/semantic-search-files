#!/bin/bash

# Script to perform a semantic search using the Python CLI

DEFAULT_MODEL_NAME="all-MiniLM-L6-v2" # Keep in sync with cli.py or make it an arg here too
MODEL_NAME="$DEFAULT_MODEL_NAME"
NUM_RESULTS=5
SHOW_SNIPPET=""
QUERY=""

# Parse arguments for this shell script
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL_NAME="$2"; shift ;;
        -k|--num-results) NUM_RESULTS="$2"; shift ;;
        --snippet) SHOW_SNIPPET="--snippet" ;;
        -h|--help)
            echo "Usage: $0 "<your search query>" [--model <model_name>] [-k <num_results>] [--snippet]"
            echo "Options:"
            echo "  "<your search query>"  The semantic query string (required)."
            echo "  --model <model_name>    Sentence Transformer model to use (default: $DEFAULT_MODEL_NAME)."
            echo "  -k, --num-results     Number of top results to retrieve (default: 5)."
            echo "  --snippet             Show a small snippet from the found files."
            echo "  -h, --help              Show this help message."
            exit 0
            ;;
        *) QUERY="$1" ;;
    esac
    shift
done

if [ -z "$QUERY" ]; then
  echo "Error: Search query is required."
  echo "Usage: $0 "<your search query>" [--model <model_name>] [-k <num_results>] [--snippet]"
  exit 1
fi

# Assuming the script is run from the project root, or adjust path to cli.py
PYTHON_CLI_PATH="src/cli.py"

if [ ! -f "$PYTHON_CLI_PATH" ]; then
    echo "Error: Python CLI script not found at $PYTHON_CLI_PATH. Make sure you are in the project root."
    exit 1
fi

echo "Performing semantic search..."
echo "Query: $QUERY"
echo "Model: $MODEL_NAME"
echo "Number of results: $NUM_RESULTS"
if [ -n "$SHOW_SNIPPET" ]; then
    echo "Show snippet: Yes"
fi

# Construct the command
COMMAND="python $PYTHON_CLI_PATH --model "$MODEL_NAME" search "$QUERY" -k $NUM_RESULTS $SHOW_SNIPPET"
echo "Executing: $COMMAND"
eval $COMMAND

echo "Search script finished."
