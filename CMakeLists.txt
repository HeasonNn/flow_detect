cmake_minimum_required(VERSION 3.10)
project(FlowAnomalyDetector)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
pkg_check_modules(MLPACK REQUIRED mlpack)

include_directories(${MLPACK_INCLUDE_DIRS})
link_directories(${MLPACK_LIBRARY_DIRS})

add_executable(flow_detector
    src/main.cpp
    src/feature/flow_feature.cpp
    src/feature/graph_features.cpp
    src/dataloader/cicids_loader.cpp
    src/dataloader/hypervison_loader.cpp
    src/flow_construct/explicit_constructor.cpp
    src/detector/rfdetector.cpp
    src/detector/detector.cpp
)

target_link_libraries(flow_detector
    ${MLPACK_LIBRARIES}
    pcap
    gomp
    pthread
    armadillo
)

target_link_libraries(flow_detector
    libPcap++.a 
    libPacket++.a 
    libCommon++.a
)
