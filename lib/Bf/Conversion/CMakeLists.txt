add_mlir_conversion_library(Passes
    Passes.cpp
    
    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Bf
    
    DEPENDS
    MLIRBfConversionPassIncGen
    
    LINK_COMPONENTS
    Core
    
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
)