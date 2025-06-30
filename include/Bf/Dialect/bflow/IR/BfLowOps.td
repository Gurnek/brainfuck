#ifndef BF_LOW_OPS
#define BF_LOW_OPS

include "Bf/Dialect/bflow/IR/BfLowDialect.td"

def Bf_read_ptr : BfLow_Op<"read_ptr"> {
    let summary = "Read current value at the pointer and return";

    let arguments = (ins SI32Attr:$offset);
    let results = (outs I16);

    let assemblyFormat = "attr-dict";
}

def Bf_write_ptr : BfLow_Op<"write_ptr"> {
    let summary = "Write current value at the pointer";

    let arguments = (ins I16:$value, SI32Attr:$offset);

    let assemblyFormat = "attr-dict-with-keyword $value";
}

def Bf_shift_ptr : BfLow_Op<"shift_ptr"> {
    let summary = "Shift pointer left or right by $value";

    let arguments = (ins SI32Attr:$amount);

    let assemblyFormat = "attr-dict";
}

#endif // BF_LOW_OPS