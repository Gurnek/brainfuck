#ifndef BF_CONVERSION_TD
#define BF_CONVERSION_TD

include "mlir/Pass/PassBase.td"

def BfCombineConsecutiveOps: Pass<"combine-consecutive-ops", "::mlir::ModuleOp"> {
    let summary = "Combines consecutive shift and increment ops into one";
    let description = [{
        bf.Increment { 1 }, bf.Increment { 1 } => bf.Increment { 2 }
    }];
}

def BfSetZero: Pass<"set-zero", "::mlir::ModuleOp"> {
    let summary = "Peephole optimization to convert ops into a single SetZero op";
    let description = [{
        Looks for bf ops in the form [-] or [+] and combines them into a single op
    }];
}

def BfOffset: Pass<"calculate-offsets", "::mlir::ModuleOp"> {
    let summary = "Calculate pointer offsets ahead of time";
    let description = [{
        If there are sequences of +-><, they can be rearranged so that the increments are done with an offset and the pointer is moved all at once after
    }];
}

def BfMulLoop: Pass<"multiply-loop", "::mlir::ModuleOp"> {
    let summary = "Peephole optimization to convert multiply loops into Mul Ops";
    let description = [{
        Looks for bf ops in the form [->+++>+++++<<] and converts them into Mul Ops
    }];
}

def BfToBfLow : Pass<"bf-to-bflow"> {
    let summary = "Lower bf dialect to a lower dialect which deals with the bf pointer";

    let description = [{
        the bf pointer operations read and write values to the location in memory that the pointer
        is pointing to. It also transfers over the shift operations into this bflow dialect.
    }];

    let dependentDialects = [
        "bf::BfDialect",
        "bflow::BfLowDialect",
        "mlir::arith::ArithDialect",
        "mlir::scf::SCFDialect",
        "mlir::BuiltinDialect",
        "mlir::LLVM::LLVMDialect",
        "mlir::func::FuncDialect",
    ];
}

def BfLowToMLIR : Pass<"bflow-to-mlir"> {
    let summary = "Lower the bf low dialect to native mlir ops";
    let description = [{
        The bflow::read_ptr op would be lowered to a load op for the global memory
    }];

    let dependentDialects = [
        "bflow::BfLowDialect",
        "mlir::index::IndexDialect",
        "mlir::memref::MemRefDialect",
        "mlir::BuiltinDialect",
        "mlir::LLVM::LLVMDialect"
    ];
}

#endif // BF_CONVERSION_TD