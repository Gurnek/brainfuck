#ifndef BF_OPS
#define BF_OPS

include "Bf/Dialect/bf/IR/BfDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/RegionKindInterface.td"


def Bf_Increment : Bf_Op<"increment"> {
    let summary = "Increment the cell by $amount";
    let arguments = (ins SI16Attr:$amount, SI32Attr:$offset);
    let assemblyFormat = "attr-dict";
}

def Bf_Shift : Bf_Op<"shift"> {
    let summary = "Shift cells by $amount";
    let arguments = (ins SI32Attr:$amount);
    let assemblyFormat = "attr-dict";
}

def Bf_Output : Bf_Op<"output"> {
    let summary = "Outputs the current cell as ASCII";
    let assemblyFormat = "attr-dict";
}

def Bf_Input : Bf_Op<"input"> {
    let summary = "Input an ASCII character to the current cell";
    let assemblyFormat = "attr-dict";
}

def Bf_Loop : Bf_Op<"loop", [DeclareOpInterfaceMethods<RegionKindInterface>, SingleBlock, NoTerminator]> {
    let summary = "Bf loop when current cell != 0";
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = "attr-dict-with-keyword $body";
}

def Bf_SetZero : Bf_Op<"set_zero"> {
    let summary = "Set current cell to 0";
    let arguments = (ins SI32Attr:$offset);
    let assemblyFormat = "attr-dict";
}

def Bf_Mul : Bf_Op<"mul"> {
    let summary = "Multiply loop optimization in an Op";
    let arguments = (ins SI32Attr:$shift, I16Attr:$multiple);
    let assemblyFormat = "attr-dict";
}

#endif //BF_OPS
