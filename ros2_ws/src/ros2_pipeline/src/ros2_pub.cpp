#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>

int main(int argc, char** argv) {
  // CPU: initialize ROS2
  rclcpp::init(argc, argv);
  // CPU: create a node named "ros2_publisher"
  auto node = rclcpp::Node::make_shared("ros2_publisher");

  // CPU: create a publisher on "ros2/input" with queue size 10
  auto pub = node->create_publisher<std_msgs::msg::Float32MultiArray>("ros2/input", 10);

  // CPU: timer callback at 10 Hz
  auto timer = node->create_wall_timer(
    std::chrono::milliseconds(100),
    [pub]() {
      std_msgs::msg::Float32MultiArray msg;
      // fill 1 048 576 floats with value 1.0f
      msg.data.assign(1 << 20, 1.0f);
      pub->publish(msg);
      RCLCPP_INFO(rclcpp::get_logger("ros2_publisher"),
                  "Published %zu floats", msg.data.size());
    }
  );

  // CPU: spin forever, invoking the timer callback
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}