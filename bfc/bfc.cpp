#include <iostream>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include "Bf/Dialect/bflow/IR/BfLowDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/AsmState.h"
#include <system_error>
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ErrorOr.h"
#include "Bf/Dialect/bf/IR/BfDialect.h"
#include "Lexer.h"
#include "MLIRGen.h"

namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<Input bf file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string> outputFilename("o",
                                           cl::desc("Output filename"),
                                           cl::value_desc("filename"),
                                           cl::init("a.mlir"));

int main(int argc, char *argv[]) {
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();

    cl::ParseCommandLineOptions(argc, argv, "Brainfuck Compiler\n");
    if (!llvm::StringRef(inputFilename).ends_with(".bf")) {
        std::cout << "Error: Must input brainfuck src file that ends with .bf.\n";
        exit(1);
    }
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }
    auto buffer = fileOrErr.get()->getBuffer();
    std::vector<bf::Token> tok_vec = bf::lex_program(buffer.str());

    mlir::DialectRegistry registry;
    registry.insert<
        mlir::func::FuncDialect,
        bf::BfDialect,
        bflow::BfLowDialect,
        mlir::BuiltinDialect
    >();

    mlir::MLIRContext context(registry);
    context.getOrLoadDialect<bf::BfDialect>();
    context.getOrLoadDialect<bflow::BfLowDialect>();
    context.loadAllAvailableDialects();

    mlir::OwningOpRef<mlir::ModuleOp> module = mlirGen(context, tok_vec);
    if (!module) {
        return 1;
    }

    std::error_code EC;
    llvm::raw_fd_ostream mlirFile(outputFilename, EC);
    if (EC) {
        llvm::errs() << "Error: Could not open file " << outputFilename << "\n";
        return 1;
    }
    module->print(mlirFile);

    return 0;
}