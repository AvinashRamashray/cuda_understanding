#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <cuda_runtime.h>
#include <cstring>  // for memcpy
#include <stdexcept> // for runtime_error

// GPU kernel: runs on the device, adds a constant to each element in parallel
__global__ void addConstantKernel(float* data, float constant, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] += constant;
}

class OptimizedCudaNode : public rclcpp::Node {
public:
  // Macro for CUDA error checking
  #define CUDA_CHECK(call) \
    do { \
      cudaError_t err = call; \
      if (err != cudaSuccess) { \
        RCLCPP_ERROR(get_logger(), "CUDA error at %s:%d code=%d(%s)", \
          __FILE__, __LINE__, err, cudaGetErrorString(err)); \
        throw std::runtime_error("CUDA failure"); \
      } \
    } while (0)

  OptimizedCudaNode()
  : Node("optimized_cuda_node"),            // CPU: initialize ROS2 node
    N_(1 << 20),                              // CPU: number of elements (1 million) //1 << 20 = 2^20 = 1,048,576 // in future it will dynamically adapt to input size
    bytes_(N_ * sizeof(float)),               // CPU: total size in bytes //1,048,576 * 4 ≈ 4 MB
    constant_(5.0f)                           // CPU: value to add in GPU kernel
  {
    RCLCPP_INFO(get_logger(), "Constructor: allocating CUDA resources");
    try {
      // 1) Preallocate pinned host buffers (faster transfers)
      CUDA_CHECK(cudaMallocHost(&h_input_,  bytes_));
      CUDA_CHECK(cudaMallocHost(&h_output_, bytes_));

      // 2) Preallocate device buffer once
      CUDA_CHECK(cudaMalloc(&d_data_, bytes_));

      // 3) Create a single CUDA stream
      CUDA_CHECK(cudaStreamCreate(&stream_));

      // 4) Create CUDA events for timing
      CUDA_CHECK(cudaEventCreate(&evt_start_));
      CUDA_CHECK(cudaEventCreate(&evt_h2d_end_));
      CUDA_CHECK(cudaEventCreate(&evt_kernel_end_));
      CUDA_CHECK(cudaEventCreate(&evt_d2h_end_));
    } catch (const std::exception &e) {
      RCLCPP_FATAL(get_logger(), "Constructor exception: %s", e.what());
      throw;
    }

    // 5) ROS2 interfaces
    RCLCPP_INFO(get_logger(), "Constructor: setting up ROS2 subscription and publisher");
    sub_ = create_subscription<std_msgs::msg::Float32MultiArray>(
      "/ros2/input", 10,
      std::bind(&OptimizedCudaNode::callback, this, std::placeholders::_1)
    );
    pub_ = create_publisher<std_msgs::msg::Float32MultiArray>(
      "/ros2/output", 10
    );
    RCLCPP_INFO(get_logger(), "Constructor completed successfully");
  }

  ~OptimizedCudaNode() noexcept {
    RCLCPP_INFO(get_logger(), "Destructor: releasing CUDA resources");
    // Cleanup without throwing
    if (cudaFreeHost(h_input_) != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "Failed to free host input buffer");
    }
    if (cudaFreeHost(h_output_) != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "Failed to free host output buffer");
    }
    if (cudaFree(d_data_) != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "Failed to free device buffer");
    }
    if (cudaStreamDestroy(stream_) != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "Failed to destroy CUDA stream");
    }
    if (cudaEventDestroy(evt_start_) != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "Failed to destroy evt_start_");
    }
    if (cudaEventDestroy(evt_h2d_end_) != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "Failed to destroy evt_h2d_end_");
    }
    if (cudaEventDestroy(evt_kernel_end_) != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "Failed to destroy evt_kernel_end_");
    }
    if (cudaEventDestroy(evt_d2h_end_) != cudaSuccess) {
      RCLCPP_ERROR(get_logger(), "Failed to destroy evt_d2h_end_");
    }
  }

private:
  void callback(const std_msgs::msg::Float32MultiArray::SharedPtr msg) {
    RCLCPP_INFO(get_logger(), "Callback: received data of size %zu", msg->data.size());
    // Copy into pinned buffer
    std::memcpy(h_input_, msg->data.data(), bytes_);

    // 1) Mark start of H2D copy
    CUDA_CHECK(cudaEventRecord(evt_start_, stream_));
    // 2) Async copy
    CUDA_CHECK(cudaMemcpyAsync(d_data_, h_input_, bytes_, cudaMemcpyHostToDevice, stream_));
    // 3) Mark end of H2D copy
    CUDA_CHECK(cudaEventRecord(evt_h2d_end_, stream_));

    // 4) Kernel launch
    const int threads = 256;
    const int blocks  = (N_ + threads - 1) / threads;
    addConstantKernel<<<blocks, threads, 0, stream_>>>(d_data_, constant_, N_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evt_kernel_end_, stream_));

    // 5) D2H copy
    CUDA_CHECK(cudaMemcpyAsync(h_output_, d_data_, bytes_, cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaEventRecord(evt_d2h_end_, stream_));

    // Wait
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // Timing
    float t_h2d=0.f, t_kernel=0.f, t_d2h=0.f;
    CUDA_CHECK(cudaEventElapsedTime(&t_h2d, evt_start_, evt_h2d_end_));
    CUDA_CHECK(cudaEventElapsedTime(&t_kernel, evt_h2d_end_, evt_kernel_end_));
    CUDA_CHECK(cudaEventElapsedTime(&t_d2h, evt_kernel_end_, evt_d2h_end_));
    RCLCPP_INFO(get_logger(), "Timings (ms): H2D=%.2f Kernel=%.2f D2H=%.2f", t_h2d, t_kernel, t_d2h);

    // Publish
    auto out = std_msgs::msg::Float32MultiArray();
    out.data.assign(h_output_, h_output_ + N_);
    pub_->publish(std::move(out));
  }

  // Members
  const size_t N_, bytes_;
  const float constant_;
  float *h_input_{nullptr}, *h_output_{nullptr}, *d_data_{nullptr};
  cudaStream_t stream_{};
  cudaEvent_t evt_start_{}, evt_h2d_end_{}, evt_kernel_end_{}, evt_d2h_end_{};
  rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr pub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);                              
  RCLCPP_INFO(rclcpp::get_logger("main"), "Starting OptimizedCudaNode");
  auto node = std::make_shared<OptimizedCudaNode>();
  rclcpp::spin(node);   
  rclcpp::shutdown();                                    
  return 0;
}

