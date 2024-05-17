cmake_minimum_required(VERSION 3.18)
project(tokenizers_cpp_example C CXX)

include(CheckCXXCompilerFlag)
if(NOT MSVC)
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
else()
  check_cxx_compiler_flag("/std:c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "/std:c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
endif()

set(TOKENZIER_CPP_PATH tokenizers-cpp)
message("Tokenizer CPP Path: ${TOKENIZER_CPP_PATH}")


add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

add_library (hanabi hanabi_card.cc hanabi_game.cc hanabi_hand.cc hanabi_history_item.cc hanabi_move.cc hanabi_observation.cc hanabi_state.cc util.cc canonical_encoders.cc)
target_include_directories(hanabi PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(hanabi PUBLIC ${TOKENZIER_CPP_PATH}/include)
target_link_libraries(hanabi PUBLIC tokenizers_cpp)

