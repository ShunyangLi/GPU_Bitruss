add_library(src
        # bfc folder
        bfc/bfc.cu
        bfc/bfc.cuh
        bfc/bfc_vpp.cu

        # bitruss folder
        bitruss/bitruss.cuh
        bitruss/bitruss_msp.cu
        bitruss/bitruss_hidx.cu

        # cpu folder
        bitruss/cpu_bitruss.cpp
        bfc/cpu_bfc.cpp

        # graph folder
        graph/graph.h
        graph/graph.cpp

        # util folder
        util/config.h
        util/dbg.h
        util/log.cpp
        util/log.h
        util/table.h
        util/timer.cuh
        util/utility.h
        util/uf.h
)

# Link libraries (if needed)
target_link_libraries(src PUBLIC argparse)

# Include directories
target_include_directories(src PUBLIC
        ${argparse_SOURCE_DIR}
        )
# target_include_directories(src PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
