#ifndef BF_PASSES_H
#define BF_PASSES_H

#include "Bf/Dialect/bf/IR/BfDialect.h"
#include "Bf/Dialect/bf/IR/BfOps.h"
#include "mlir/Pass/Pass.h"

namespace bf {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Bf/Conversion/Passes.h.inc"
} // namespace bf



#endif // BF_PASSES_H