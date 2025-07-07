# Brainfuck Compiler

An optimizing Brainfuck compiler written in MLIR. The bfc.sh script is the entrypoint that invokes several binaries to convert brainfuck source files to executables.

## Dialects

This compiler consists of two main dialects. The first dialect is a high level dialect that directly maps the operations in the Brainfuck language to Ops in the dialect.
There are the following ops:

- Increment

- Shift

- Output

- Input

- Loop

- SetZero

- Multiply

The first 5 correspond to the base operations in the Brainfuck language, while SetZero and Multiply are generated in an optimization pass and represent typical operations in Brainfuck programs.

The next dialect is lower level in nature, and deals directly with reading and writing from the internal buffer and the pointer object.
There are the following ops:

- ReadPtr

- WritePtr

- ShiftPtr

These operations will read/write to the memory location the pointer points to, or they can shift the pointer itself. All of the ops from the previous dialect are lowered to ops from standard MLIR dialects
and ops in this dialect. For instance increment will ReadPtr from memory, increment the value, and then WritePtr that value back into memory. The other operations are similar.

After this, the Ops are lowered to standard MLIR dialects and then the inbuilt passes are used to lower it to LLVMIR. From there the LLVM compiler is invoked to compile it to an object and executable.
