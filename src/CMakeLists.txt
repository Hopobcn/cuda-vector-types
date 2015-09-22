include_directories("../include")

cuda_add_library(libgpu STATIC vectorAdd.cu
                 OPTIONS
                 DEBUG          "-std=c++11 -g -G -O0 -lineinfo -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 "
                 RELEASE        "-std=c++11 -g    -O3 -lineinfo -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 --ptxas-options -v -keep"
                 MINSIZEREL     "-std=c++11 -g    -O3 -lineinfo -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 "
                 RELWITHDEBINFO "-std=c++11 -g    -O3 -lineinfo -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 " )

cuda_add_executable(cuda-vector-types main.cpp)
target_link_libraries(cuda-vector-types libgpu)