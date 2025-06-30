get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS 
        ${dialect_libs}
        ${conversion_libs}
        MLIRIR
        MLIRArithDialect
        MLIRFuncDialect
        MLIRPass
        MLIRMemRefDialect
        MLIROptLib
        MLIRBf
        MLIRBfLow
        )

set(LLVM_LINK_COMPONENTS Support)
add_llvm_executable(bfc bfc.cpp)
llvm_update_compile_flags(bfc)
target_link_libraries(bfc PRIVATE ${LIBS})

mlir_check_all_link_libraries(bfc)