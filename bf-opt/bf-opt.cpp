#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Bf/Dialect/bf/IR/BfDialect.h"
#include "Bf/Dialect/bflow/IR/BfLowDialect.h"
#include "Bf/Conversion/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Index/IR/IndexDialect.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

int main(int argc, char *argv[]) {
    mlir::registerAllPasses();
    bf::registerPasses();

    mlir::DialectRegistry registry;
    registry.insert<bf::BfDialect>();
    registry.insert<bflow::BfLowDialect>();
    registry.insert<
        mlir::index::IndexDialect, 
        mlir::arith::ArithDialect, 
        mlir::func::FuncDialect, 
        mlir::memref::MemRefDialect, 
        mlir::scf::SCFDialect, 
        mlir::cf::ControlFlowDialect,
        mlir::LLVM::LLVMDialect
    >();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Bf optimizer driver\n", registry)
    );
}