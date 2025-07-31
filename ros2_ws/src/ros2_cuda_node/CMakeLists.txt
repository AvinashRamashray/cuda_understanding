cmake_minimum_required(VERSION 3.10)
project(ros2_cuda_node)

find_package(ament_cmake REQUIRED)
find_package(rclcpp       REQUIRED)
find_package(std_msgs     REQUIRED)
find_package(CUDA         REQUIRED)

# optional: set optimization flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

enable_language(CUDA)

add_executable(cuda_node
  src/ros2_cuda_node.cu           # or ros2_cuda_optimized.cu
)

ament_target_dependencies(cuda_node
  rclcpp std_msgs
)

# Tell CMake this target uses CUDA
set_target_properties(cuda_node PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

install(TARGETS cuda_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
