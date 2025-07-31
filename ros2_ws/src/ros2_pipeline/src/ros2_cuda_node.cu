#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <cuda_runtime.h>
#include <chrono>

__global__ void addConstant(float* data, float constant, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] += constant;
}

class CudaNode : public rclcpp::Node {
public:
  CudaNode() : Node("cuda_node") {
    sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      "/ros2/input", 10,
      std::bind(&CudaNode::callback, this, std::placeholders::_1));
    pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      "/ros2/output", 10);
  }
//std_msgs::msg::Float32MultiArray
private:
  void callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
    size_t N = msg->data.size();  //1 048 576 (1 million) data points
    //msg->data[i] == 1.0f   // for every i from 0 to 1 048 575
    size_t bytes = N * sizeof(float); //1048576 x 4 = 4194304 bytes = 4 MiB

    // 1) Allocate device memory *every callback*
    float* d_data;
    cudaMalloc(&d_data, bytes);  // Allocate 4 MiB on the GPU

    // 2) Time from first copy to final copy
    auto start = std::chrono::high_resolution_clock::now();

    // 3) Host→Device
    cudaMemcpy(d_data, msg->data.data(), bytes, cudaMemcpyHostToDevice); //Copy the entire array to GPU global memory:
    //— now on the GPU, d_data[i] == 1.0f for 0 ≤ i < 1 048 576.

    // 4) Kernel launch
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;//// (1 048 576 + 255) / 256 = 4 096 blocks
    addConstant<<<blocks, threads>>>(d_data, 5.0f, N);//— that fires 4 096 blocks, each with 256 threads, for a total of 1 048 576 threads.
    //addConstant<<< 4096, 256 >>>(d_data, 5.0f, N);

    cudaDeviceSynchronize();//// waits until all 1 048 576 updates are done

    // 5) Device→Host
    std::vector<float> output(N);
    cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost);
    //— copying back 4 MiB of floats from GPU to CPU. Now in your host output[i] == 6.0f.

    auto end = std::chrono::high_resolution_clock::now();
    auto ms  = std::chrono::duration<double, std::milli>(end - start).count();
    RCLCPP_INFO(get_logger(), "GPU processing took %.2f ms", ms);

    cudaFree(d_data);

    // 6) Publish result
    auto out_msg = std_msgs::msg::Float32MultiArray();
    out_msg.data = std::move(output);
    pub_->publish(std::move(out_msg));
  }

  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CudaNode>());
  rclcpp::shutdown();
  return 0;
}