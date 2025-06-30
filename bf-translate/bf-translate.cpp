#include "Bf/Dialect/bf/IR/BfDialect.h"
#include "Bf/Dialect/bflow/IR/BfLowDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Operation.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "llvm/Support/raw_ostream.h"
#include <llvm/Support/LogicalResult.h>

int main(int argc, char *argv[]) {
    mlir::registerAllTranslations();

    mlir::TranslateFromMLIRRegistration withdescription(
        "option", "different from option",
        [](mlir::Operation *op, llvm::raw_ostream &output) {
            return llvm::LogicalResult::success();
        },
        [](mlir::DialectRegistry &a) {}
    );

    return llvm::failed(mlir::mlirTranslateMain(argc, argv, "Bf translation tool"));
}