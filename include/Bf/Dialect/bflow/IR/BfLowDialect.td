#ifndef BF_LOW_DIALECT
#define BF_LOW_DIALECT

include "mlir/IR/OpBase.td"

def BfLow_Dialect : Dialect {
    let name = "BfLow";
    let summary = "Lowered Bf operations on the memory and pointer for brainfuck";
    let cppNamespace = "bflow";
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

class BfLow_Op<string mnemonic, list<Trait> traits = []> : Op<BfLow_Dialect, mnemonic, traits>;

#endif //BF_LOW_DIALECT