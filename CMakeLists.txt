cmake_minimum_required(VERSION 3.10)
project(FlowAnomalyDetector)

set(CMAKE_CXX_STANDARD 17)

# 设置 Debug 构建类型为默认（可选）
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# 设置编译器选项
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

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
    src/dataloader/data_loader.cpp
    src/dataloader/pcap_loader.cpp
    src/flow_construct/explicit_constructor.cpp
    src/detector/detector.cpp
    src/detector/rf_detector.cpp
    src/detector/dbscan_detector.cpp
    src/detector/iforest_detector.cpp
    src/detector/mbk_detector.cpp
    src/algorithm/mini_batch_kmeans.cpp
    src/graph_construct/edge_constructor.cpp
    src/parser/packet_parser.cpp
)

target_link_libraries(flow_detector
    ${MLPACK_LIBRARIES}
    gflags
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
