#!/bin/bash

set -e

OPT_LEVEL=0
DEBUG=false

usage() {
    echo "Usage: $0 [options] <input.mlir> <output_executable>"
    echo "Options:"
    echo "  -O <level>  Optimization level (0-3), default: 0"
    echo "  -debug      Keep intermediate files, default: false"
    echo "  -h, --help  Show this help message"
    exit 1
}

POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -O)
            OPT_LEVEL="$2"
            if [[ ! "$OPT_LEVEL" =~ ^[0-3]$ ]]; then
                echo "Error: Invalid optimization level. Must be 0-3."
                exit 1
            fi
            shift 2
            ;;
        -debug)
            DEBUG=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}"

if [ $# -ne 2 ]; then
    usage
fi

INPUT_FILE=$1
OUTPUT_FILE=$2

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file $INPUT_FILE not found."
    exit 1
fi

TEMP_DIR=$(mktemp -d)
BASE_NAME=$(basename "$OUTPUT_FILE")
MLIR_FILE="$TEMP_DIR/${BASE_NAME}.mlir"
MLIR_LOW_FILE="$TEMP_DIR/${BASE_NAME}_low.mlir"
LLVM_FILE="$TEMP_DIR/${BASE_NAME}.ll"
OPT_FILE="$TEMP_DIR/${BASE_NAME}.opt.ll"
OBJ_FILE="$TEMP_DIR/${BASE_NAME}.o"

cleanup() {
    if [ "$DEBUG" = false ]; then
        rm -rf "$TEMP_DIR"
    else
        echo "Debug mode: Intermediate files kept in $TEMP_DIR"
    fi
}

trap cleanup EXIT

echo "Compiling $INPUT_FILE to $OUTPUT_FILE with optimization level $OPT_LEVEL..."

echo "Running bfc (.bf -> .mlir)..."
./build/bin/bfc -o "$MLIR_FILE" "$INPUT_FILE"

echo "Running bf-opt (.mlir -> .mlir)..."
PASSES=""
if [ "$OPT_LEVEL" -ge 1 ]; then
    PASSES+="-combine-consecutive-ops -set-zero -calculate-offsets -multiply-loop "
fi
PASSES+="-bf-to-bflow -bflow-to-mlir "
PASSES+="-convert-scf-to-cf -convert-cf-to-llvm -convert-index-to-llvm -convert-arith-to-llvm -convert-func-to-llvm -expand-strided-metadata -finalize-memref-to-llvm "
PASSES+="-reconcile-unrealized-casts -canonicalize -cse"
./build/bin/bf-opt -o "$MLIR_LOW_FILE" $PASSES "$MLIR_FILE"

echo "Running bf-translate (.mlir -> .ll)"
./build/bin/bf-translate --mlir-to-llvmir -o "$LLVM_FILE" "$MLIR_LOW_FILE"

echo "Running opt (LLVM optimizations)"
if [ "$OPT_LEVEL" -le 1 ]; then
    OPT_FILE=$LLVM_FILE
else
    opt -S -O"$OPT_LEVEL" "$LLVM_FILE" -o "$OPT_FILE"

    if [ "$OPT_LEVEL" -ge 2 ]; then
        opt -S -O"$OPT_LEVEL" "$OPT_FILE" -o "$OPT_FILE"
    fi
fi

echo "Running llc (.ll -> .o)"
llc -filetype=obj --relocation-model=pic "$OPT_FILE" -o "$OBJ_FILE"

echo "Running clang (linking)"
clang -O"$OPT_LEVEL" "$OBJ_FILE" -o "$OUTPUT_FILE"

echo "Successfully compiled $INPUT_FILE to $OUTPUT_FILE"
echo "Done!"
