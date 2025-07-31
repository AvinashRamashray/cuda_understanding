#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <cuda_runtime.h>

__global__ void addConstantKernel(float* data, float constant, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] += constant;
}

class OptimizedCudaNode : public rclcpp::Node {
public:
  OptimizedCudaNode()
  : Node("optimized_cuda_node"),
    N_(1 << 20),
    bytes_(N_ * sizeof(float)),
    constant_(5.0f)
  {
    // 1) Preallocate pinned host buffers (faster transfers)
    cudaMallocHost(&h_input_,  bytes_);
    cudaMallocHost(&h_output_, bytes_);

    // 2) Preallocate device buffer once
    cudaMalloc(&d_data_, bytes_);

    // 3) Create a single CUDA stream
    cudaStreamCreate(&stream_);

    // 4) Create CUDA events for timing
    cudaEventCreate(&evt_h2d_);
    cudaEventCreate(&evt_kernel_);
    cudaEventCreate(&evt_d2h_);

    // 5) ROS2 interfaces
    sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      "/ros2/input", 1,
      std::bind(&OptimizedCudaNode::callback, this, std::placeholders::_1));
    pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      "/ros2/output", 1);
  }

  ~OptimizedCudaNode() {
    cudaFreeHost(h_input_);
    cudaFreeHost(h_output_);
    cudaFree(d_data_);
    cudaStreamDestroy(stream_);
    cudaEventDestroy(evt_h2d_);
    cudaEventDestroy(evt_kernel_);
    cudaEventDestroy(evt_d2h_);
  }

private:
  void callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
    // Copy into pinned buffer
    memcpy(h_input_, msg->data.data(), bytes_);

    // H2D
    cudaEventRecord(evt_h2d_, stream_);
    cudaMemcpyAsync(d_data_, h_input_, bytes_, cudaMemcpyHostToDevice, stream_);

    // Kernel
    cudaEventRecord(evt_kernel_, stream_);
    const int threads = 256;
    const int blocks  = (N_ + threads - 1) / threads;
    addConstantKernel<<<blocks, threads, 0, stream_>>>(d_data_, constant_, N_);

    // D2H
    cudaEventRecord(evt_d2h_, stream_);
    cudaMemcpyAsync(h_output_, d_data_, bytes_, cudaMemcpyDeviceToHost, stream_);

    // Wait once
    cudaStreamSynchronize(stream_);

    // Read timings
    float t_h2d = 0.f, t_kern = 0.f, t_d2h = 0.f;
    cudaEventElapsedTime(&t_h2d,  evt_h2d_,  evt_kernel_);
    cudaEventElapsedTime(&t_kern, evt_kernel_, evt_d2h_);
    cudaEventElapsedTime(&t_d2h,  evt_d2h_,  evt_d2h_);
    RCLCPP_INFO(get_logger(),
      "Timings (ms): H2D=%.2f  Kernel=%.2f  D2H=%.2f",
      t_h2d, t_kern, t_d2h);

    // Publish result
    auto out = std_msgs::msg::Float32MultiArray();
    out.data.assign(h_output_, h_output_ + N_);
    pub_->publish(std::move(out));
  }

  // Members
  size_t N_, bytes_;
  float  constant_;
  float *h_input_, *h_output_, *d_data_;
  cudaStream_t stream_;
  cudaEvent_t   evt_h2d_, evt_kernel_, evt_d2h_;
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OptimizedCudaNode>());
  rclcpp::shutdown();
  return 0;
}