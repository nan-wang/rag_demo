#!/bin/bash

# --- Script Description ---
# This script runs a sequence of Python scripts to generate QA pairs,
# validate, rewrite, extract keypoints
#

# --- Usage Function ---
usage() {
    echo "Usage: $0 --index-dir <path> --collection-name <name> --output-dir <path> --num-generate <int> --num-metric <int>"
    echo ""
    echo "Options:"
    echo "  --index-dir <path>      : Directory containing the document index (e.g., ./data_chroma_multi)"
    echo "  --collection-name <name>: Name of the collection within the index (e.g., test_db)"
    echo "  --output-dir <path>     : Base directory where all synthetic data will be stored (e.g., data_eval/v20250501)"
    echo "  --num-generate <int>    : Number of synthetic QA pairs to generate (e.g., 128)"
    echo "  -h, --help              : Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --index-dir ./data_chroma_multi --collection-name test_db --output-dir ./eval_eval/v20250506 --num-generate 256"
    echo "  $0 --index-dir ./my_index --collection-name my_collection --output-dir ./eval_results_20231231 --num-generate 200"
}

# --- Parse Arguments ---
# Set initial values to empty - we'll require all of them
INDEX_DIR=""
COLLECTION_NAME=""
BASE_OUTPUT_DIR=""
NUM_GENERATE_DOCS=""

# Loop through arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --index-dir) INDEX_DIR="$2"; shift ;;
        --collection-name) COLLECTION_NAME="$2"; shift ;;
        --output-dir) BASE_OUTPUT_DIR="$2"; shift ;;
        --num-generate) NUM_GENERATE_DOCS="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Error: Unknown parameter passed: $1"; usage; exit 1 ;;
    esac
    shift # Consume the argument (either option or value if no shift happened inside case)
done

# --- Validate Arguments ---
if [ -z "$INDEX_DIR" ] || [ -z "$COLLECTION_NAME" ] || [ -z "$BASE_OUTPUT_DIR" ] || [ -z "$NUM_GENERATE_DOCS" ] ; then
    echo "Error: Missing required arguments."
    usage
    exit 1
fi

# --- Set Derived Variables and Create Directories ---
RAW_OUTPUT_PATH="${BASE_OUTPUT_DIR}/qa_pairs.raw.json"
VALIDATED_OUTPUT_PATH="${BASE_OUTPUT_DIR}/qa_pairs.validated.json"
REWRITTEN_OUTPUT_PATH="${BASE_OUTPUT_DIR}/qa_pairs.rewritten.json"
KEYPOINTS_OUTPUT_PATH="${BASE_OUTPUT_DIR}/keypoints.json"


# Create necessary output directories
echo "Creating output directories: $BASE_OUTPUT_DIR"
mkdir -p "$BASE_OUTPUT_DIR" || { echo "Error: Failed to create directories."; exit 1; }

# --- Execute Steps ---

# Step 1: generate Question-Answer pairs
echo "--- Step 1: Generating QA pairs ---"
python 03_generate_qa_pairs.py \
    --index_dir "$INDEX_DIR" \
    --collection_name "$COLLECTION_NAME" \
    --output_dir "$BASE_OUTPUT_DIR" \
    --num_docs "$NUM_GENERATE_DOCS" || { echo "Error during Step 1. Aborting."; exit 1; }

# Step 2: validate the Question-Answer pairs
echo "--- Step 2: Validating QA pairs ---"
python 03_validate_qa_pairs.py \
    --input_path "${RAW_OUTPUT_PATH}" \
    --output_path "${VALIDATED_OUTPUT_PATH}" || { echo "Error during Step 2. Aborting."; exit 1; }

# Step 3: rewrite the Questions
echo "--- Step 3: Rewriting Questions ---"
python 03_rewrite_qa_pairs.py \
    --input_path "${VALIDATED_OUTPUT_PATH}" \
    --output_path "${REWRITTEN_OUTPUT_PATH}" || { echo "Error during Step 3. Aborting."; exit 1; }

# Step 4: extract the keypoints for the groundtruth
echo "--- Step 4: Extracting keypoints for groundtruth ---"
python 01_extract_keypoints.py \
    --num_docs "$NUM_GENERATE_DOCS" \
    --groundtruth \
    --input_path "${REWRITTEN_OUTPUT_PATH}" \
    --output_path "${KEYPOINTS_OUTPUT_PATH}" || { echo "Error during Step 4. Aborting."; exit 1; }

# Step 5: filter the keypoints that are not related to the question

echo "--- All steps completed successfully! ---"

exit 0