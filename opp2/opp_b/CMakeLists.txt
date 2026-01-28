cmake_minimum_required(VERSION 3.22)
project(opp2b)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_STANDARD 20)

add_executable(opp2b lab2b.c)
target_link_libraries(opp2b PRIVATE OpenMP::OpenMP_C m)
add_custom_target(run
        COMMAND ${CMAKE_PROJECT_NAME}
        COMMAND ${CMAKE_PROJECT_NAME}
        COMMAND ${CMAKE_PROJECT_NAME}
        COMMAND ${CMAKE_PROJECT_NAME}
        COMMAND ${CMAKE_PROJECT_NAME}
        )
