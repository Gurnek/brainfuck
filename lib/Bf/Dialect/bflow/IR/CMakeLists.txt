add_mlir_dialect_library(MLIRBfLow
        BfLowDialect.cpp
        BfLowOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Bf/Dialect/bflow

        DEPENDS
        MLIRBfLowOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )