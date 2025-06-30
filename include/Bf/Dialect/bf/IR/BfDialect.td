#ifndef BF_DIALECT
#define BF_DIALECT

include "mlir/IR/OpBase.td"

def Bf_Dialect : Dialect {
    let name = "Bf";
    let summary = "MLIR dialect for the Brainfuck language";
    let cppNamespace = "bf";
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

class Bf_Op<string mnemonic, list<Trait> traits = []> : Op<Bf_Dialect, mnemonic, traits>;

#endif //BF_DIALECT