#include <llvm/ADT/ArrayRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Region.h>
#include <vector>

#include "Bf/Dialect/bf/IR/BfOps.h"
#include "Lexer.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

size_t mlirGenHelper(mlir::OpBuilder &builder, std::vector<bf::Token> &tok_vec, size_t start);

mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context, std::vector<bf::Token> &tok_vec) {
    mlir::OpBuilder builder(&context);
    mlir::ModuleOp theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(theModule.getBody());
    auto main = builder.create<mlir::func::FuncOp>(
        builder.getUnknownLoc(),
        "main",
        builder.getFunctionType(mlir::TypeRange(), mlir::TypeRange()),
        llvm::ArrayRef<mlir::NamedAttribute>(),
        llvm::ArrayRef<mlir::DictionaryAttr>()
    );
    auto entry_block = main.addEntryBlock();
    builder.setInsertionPointToStart(entry_block);
    mlirGenHelper(builder, tok_vec, 0);
    builder.setInsertionPointToEnd(entry_block);
    builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());
    if (llvm::failed(mlir::verify(theModule))) {
        theModule.emitError("module verification error");
        return nullptr;
    }

    return theModule;
}

size_t mlirGenHelper(mlir::OpBuilder &builder, std::vector<bf::Token> &tok_vec, size_t start) {
    auto location = builder.getUnknownLoc();
    for (size_t i = start; i < tok_vec.size(); ++i) {
        bf::Token &tok = tok_vec[i];

        switch (tok) {
            case bf::shift_right:
                builder.create<bf::Shift>(location, 1);
                break;
            case bf::shift_left:
                builder.create<bf::Shift>(location, -1);
                break;
            case bf::increment:
                builder.create<bf::Increment>(location, 1, 0);
                break;
            case bf::decrement:
                builder.create<bf::Increment>(location, -1, 0);
                break;
            case bf::output:
                builder.create<bf::Output>(location);
                break;
            case bf::input:
                builder.create<bf::Input>(location);
                break;
            case bf::jumpz: {
                bf::Loop loop = builder.create<bf::Loop>(location);
                auto p = builder.getInsertionPoint();
                auto block = builder.getInsertionBlock();
                mlir::Block &loopBlock = loop.getBody().emplaceBlock();
                builder.setInsertionPointToStart(&loopBlock);
                i = mlirGenHelper(builder, tok_vec, i + 1);
                builder.setInsertionPoint(block, p);
                break;
            }
            case bf::jumpnz:
                return i;
        }
    }
    return tok_vec.size();
}
