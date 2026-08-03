cmake_minimum_required(VERSION 3.16)
project(PacketAnalyzer VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Header files location
include_directories(${CMAKE_SOURCE_DIR}/include)

# Common sources
set(COMMON_SOURCES
    src/pcap_reader.cpp
    src/packet_parser.cpp
    src/sni_extractor.cpp
    src/types.cpp
    src/ml_feature_exporter.cpp
    src/ml_classifier.cpp
)

# Core targets
add_executable(packet_analyzer src/main.cpp ${COMMON_SOURCES})
add_executable(dpi_working src/main_working.cpp ${COMMON_SOURCES})
add_executable(dpi_mt src/dpi_mt.cpp ${COMMON_SOURCES})

# Windows socket library linking
if(WIN32)
    target_link_libraries(packet_analyzer ws2_32)
    target_link_libraries(dpi_working ws2_32)
    target_link_libraries(dpi_mt ws2_32)
endif()

