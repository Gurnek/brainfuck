set(LLVM_LINK_COMPONENTS
    Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_executable(bf-translate bf-translate.cpp)
llvm_update_compile_flags(bf-translate)
target_link_libraries(bf-translate
    PRIVATE
    ${dialect_libs}
    ${translation_libs}
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRTranslateLib
    MLIRSupport
)

mlir_check_link_libraries(bf-translate)