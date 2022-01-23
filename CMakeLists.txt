cmake_minimum_required (VERSION 3.8)

add_executable (NeuralNet "Main.c" "NeuralNetwork.c")

add_custom_target(run
    COMMAND NeuralNet
    DEPENDS NeuralNet
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)

add_compile_options(-O3)