cmake_minimum_required(VERSION 3.10)
project(ros2_pipeline)

find_package(ament_cmake REQUIRED)
find_package(rclcpp       REQUIRED)
find_package(std_msgs     REQUIRED)
find_package(CUDA         REQUIRED)

# Target architectures—adjust '75 87 89' to your GPU
set(CMAKE_CUDA_ARCHITECTURES 75 87 89)
enable_language(CUDA)

# Publisher (CPU-only)
add_executable(ros2_pub src/ros2_pub.cpp)
ament_target_dependencies(ros2_pub rclcpp std_msgs)

# CUDA node
#add_executable(cuda_node src/ros2_cuda_node.cu)


# Optimized CUDA node
add_executable(cuda_node src/ros2_cuda_optimized.cu)

set_target_properties(cuda_node PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
ament_target_dependencies(cuda_node rclcpp std_msgs CUDA)

install(TARGETS ros2_pub cuda_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()

