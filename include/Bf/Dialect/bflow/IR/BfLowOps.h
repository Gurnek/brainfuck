#ifndef Bf_BfLowOPS_H
#define Bf_BfLowOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#define GET_OP_CLASSES
#include "Bf/Dialect/bflow/IR/BfLowOps.h.inc"

#endif