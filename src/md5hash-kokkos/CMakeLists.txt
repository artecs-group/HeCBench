cmake_minimum_required(VERSION 3.20.5)

set(This md5hash)

project(${This} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(supported_devices
    cpu
    igpu # intel GPU
    ngpu # nvidia GPU
)

if(DEVICE)
    list(FIND supported_devices ${DEVICE} device_idx)
    if(device_idx EQUAL -1)
        message(FATAL_ERROR "Please specify DEVICE (one of: ${supported_devices})")
    endif()
else()
    set(DEVICE cpu)
endif()

if(DEVICE STREQUAL "igpu")
    set(DEV_TYPE "INTEL_GPU")
    message(STATUS "Chosen Intel GPU as the device to run.")
elseif(DEVICE STREQUAL "ngpu")
    find_package(CUDA REQUIRED)
    set(DEV_TYPE "NVIDIA_GPU")
    message(STATUS "Chosen NVIDIA GPU as the device to run.")
else() # CPU
    set(DEV_TYPE "X86")
    message(STATUS "Chosen CPU as the device to run.")
endif()

find_package(Kokkos REQUIRED)
find_package(oneDPL REQUIRED)

if(NOT DEFINED ENV{KOKKOS_INSTALL_DIR})
    set(KOKKOS_INSTALL_DIR "/opt/kokkos/build/" CACHE PATH "Path to which Kokkos has been installed") 
endif()

set(CMAKE_CXX_EXTENSIONS Off)
if(DEVICE STREQUAL "igpu")
    #set(Kokkos_ARCH_INTEL_GEN9 ON)
    set(Kokkos_ENABLE_CXX11_DISPATCH_LAMBDA ON)
    set(CMAKE_CXX_COMPILER icpx)
else()
    set(Kokkos_ENABLE_CUDA_LAMBDA ON)
    #set(Kokkos_ARCH_PASCAL61 ON)
    set(Kokkos_ENABLE_CXX11_DISPATCH_LAMBDA ON)
    set(CMAKE_CXX_COMPILER ${KOKKOS_INSTALL_DIR}/bin/nvcc_wrapper)
endif()

set(Kokkos_ROOT ${KOKKOS_INSTALL_DIR}/lib/cmake/Kokkos)

set(Sources
    MD5Hash.cpp
)

set(Headers
)

set(CFLAGS "-O3")
add_executable(${This} ${Sources} ${Headers})
if(DEVICE STREQUAL "igpu")
    target_include_directories(${This} PRIVATE ${oneDPL_INCLUDE_DIRS})
endif()
add_definitions(-D"${DEV_TYPE}")
target_compile_options(${This} PRIVATE ${CFLAGS})
target_link_libraries(${This} Kokkos::kokkos)
if(DEVICE STREQUAL "igpu")
    target_link_libraries(${This} ${oneDPL_LIBRARIES})
endif()


add_custom_target(run
    COMMAND ./${This} 1 1
    DEPENDS ${PROJECT_BINARY_DIR}/${This}
    WORKING_DIRECTORY .
)