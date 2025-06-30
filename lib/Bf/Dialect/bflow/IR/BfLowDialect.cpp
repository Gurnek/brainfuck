#include "Bf/Dialect/bflow/IR/BfLowDialect.h"
#include "Bf/Dialect/bflow/IR/BfLowOps.h"

using namespace bflow;

#include "Bf/Dialect/bflow/IR/BfLowOpsDialect.cpp.inc"

void BfLowDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "Bf/Dialect/bflow/IR/BfLowOps.cpp.inc"
            >();
}