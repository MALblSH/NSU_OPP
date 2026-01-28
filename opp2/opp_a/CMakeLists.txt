cmake_minimum_required(VERSION 3.22)
project(opp2)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_STANDARD 20)

add_executable(opp2 lab2a.c)
target_link_libraries(opp2 PRIVATE OpenMP::OpenMP_C m)
add_custom_target(run
        COMMAND ${CMAKE_PROJECT_NAME}
        COMMAND ${CMAKE_PROJECT_NAME}
        COMMAND ${CMAKE_PROJECT_NAME}
        COMMAND ${CMAKE_PROJECT_NAME}
        COMMAND ${CMAKE_PROJECT_NAME}
        )

