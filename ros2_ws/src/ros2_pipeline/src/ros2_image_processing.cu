// cuda_ros_image_pipeline.cpp
// Example ROS2 node integrating CUDA-based image processing (grayscale -> Gaussian blur -> Sobel edge detection)
// Illustrates pinned host buffers, async copies, streams, and multiple CUDA kernels

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      RCLCPP_ERROR(rclcpp::get_logger("OptimizedImageNode"),              \
        "CUDA error at %s:%d code=%d(%s)",                               \
        __FILE__, __LINE__, err, cudaGetErrorString(err));                 \
      throw std::runtime_error("CUDA failure");                          \
    }                                                                      \
  } while (0)

// Kernel 1: RGB -> Grayscale
__global__ void rgb2grayKernel(const unsigned char* rgb, unsigned char* gray, int width, int height, int step) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int idx = y * step + 3 * x;
    unsigned char r = rgb[idx + 0];
    unsigned char g = rgb[idx + 1];
    unsigned char b = rgb[idx + 2];
    gray[y * width + x] = static_cast<unsigned char>(0.299f*r + 0.587f*g + 0.114f*b);
  }
}

// Kernel 2: Simple 3x3 Gaussian blur (ignores borders)
__global__ void gaussianBlurKernel(const unsigned char* gray, unsigned char* blur, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x > 0 && x < width-1 && y > 0 && y < height-1) {
    int idx = y * width + x;
    // 3x3 kernel weights
    float sum = 0;
    sum += gray[idx - width - 1] * 1;
    sum += gray[idx - width    ] * 2;
    sum += gray[idx - width + 1] * 1;
    sum += gray[idx - 1        ] * 2;
    sum += gray[idx            ] * 4;
    sum += gray[idx + 1        ] * 2;
    sum += gray[idx + width - 1] * 1;
    sum += gray[idx + width    ] * 2;
    sum += gray[idx + width + 1] * 1;
    blur[idx] = static_cast<unsigned char>(sum / 16.0f);
  }
}

// Kernel 3: Sobel edge detection
__global__ void sobelKernel(const unsigned char* blur, unsigned char* edges, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x > 0 && x < width-1 && y > 0 && y < height-1) {
    int idx = y * width + x;
    int gx = -blur[idx - width - 1] - 2*blur[idx - 1] - blur[idx + width - 1]
             + blur[idx - width + 1] + 2*blur[idx + 1] + blur[idx + width + 1];
    int gy = -blur[idx - width - 1] - 2*blur[idx - width] - blur[idx - width + 1]
             + blur[idx + width - 1] + 2*blur[idx + width] + blur[idx + width + 1];
    edges[idx] = static_cast<unsigned char>(min(255, (abs(gx) + abs(gy))));
  }
}

class OptimizedImageNode : public rclcpp::Node {
public:
  OptimizedImageNode()
  : Node("optimized_image_node")
  {
    // Subscribe to raw image (RGB8)
    sub_ = create_subscription<sensor_msgs::msg::Image>(
      "/camera/image_raw", 1,
      std::bind(&OptimizedImageNode::callback, this, std::placeholders::_1));
    pub_ = create_publisher<sensor_msgs::msg::Image>(
      "/camera/image_processed", 1);

    // Initialize CUDA stream and events
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUDA_CHECK(cudaEventCreate(&evt_start_));
    CUDA_CHECK(cudaEventCreate(&evt_done_));
  }

  ~OptimizedImageNode() {
    cudaFreeHost(h_rgb_);
    cudaFreeHost(h_gray_);
    cudaFreeHost(h_blur_);
    cudaFreeHost(h_edges_);
    cudaFree(d_rgb_);
    cudaFree(d_gray_);
    cudaFree(d_blur_);
    cudaFree(d_edges_);
    cudaStreamDestroy(stream_);
    cudaEventDestroy(evt_start_);
    cudaEventDestroy(evt_done_);
  }

private:
  void callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    int width  = msg->width;
    int height = msg->height;
    int step   = msg->step;  // bytes per row = width*3
    size_t rgb_bytes   = step * height;
    size_t gray_bytes  = width * height;

    // Allocate once on first callback
    if (!h_rgb_) {
      CUDA_CHECK(cudaMallocHost(&h_rgb_,   rgb_bytes));
      CUDA_CHECK(cudaMallocHost(&h_gray_,  gray_bytes));
      CUDA_CHECK(cudaMallocHost(&h_blur_,  gray_bytes));
      CUDA_CHECK(cudaMallocHost(&h_edges_, gray_bytes));

      CUDA_CHECK(cudaMalloc(&d_rgb_,   rgb_bytes));
      CUDA_CHECK(cudaMalloc(&d_gray_,  gray_bytes));
      CUDA_CHECK(cudaMalloc(&d_blur_,  gray_bytes));
      CUDA_CHECK(cudaMalloc(&d_edges_, gray_bytes));
    }

    // Copy incoming image to pinned host buffer (page-locked)
    memcpy(h_rgb_, msg->data.data(), rgb_bytes);

    // Record start
    cudaEventRecord(evt_start_, stream_);

    // Async copy RGB to device
    cudaMemcpyAsync(d_rgb_, h_rgb_, rgb_bytes, cudaMemcpyHostToDevice, stream_);

    // Launch rgb2gray kernel
    dim3 threads(16, 16);
    dim3 blocks((width+15)/16, (height+15)/16);
    rgb2grayKernel<<<blocks, threads, 0, stream_>>>(d_rgb_, d_gray_, width, height, step);

    // Gaussian blur
    gaussianBlurKernel<<<blocks, threads, 0, stream_>>>(d_gray_, d_blur_, width, height);

    // Sobel edge detection
    sobelKernel<<<blocks, threads, 0, stream_>>>(d_blur_, d_edges_, width, height);

    // Async copy edges back to host
    cudaMemcpyAsync(h_edges_, d_edges_, gray_bytes, cudaMemcpyDeviceToHost, stream_);

    // Wait for completion
    cudaStreamSynchronize(stream_);
    cudaEventRecord(evt_done_, stream_);

    // Publish edge image
    auto out = sensor_msgs::msg::Image();
    out.header   = msg->header;
    out.height   = height;
    out.width    = width;
    out.encoding = "mono8";
    out.is_bigendian = msg->is_bigendian;
    out.step     = width;
    out.data.assign(h_edges_, h_edges_ + gray_bytes);
    pub_->publish(std::move(out));
  }

  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;

  // CUDA buffers (host pinned and device)
  unsigned char *h_rgb_{nullptr}, *h_gray_{nullptr}, *h_blur_{nullptr}, *h_edges_{nullptr};
  unsigned char *d_rgb_{nullptr}, *d_gray_{nullptr}, *d_blur_{nullptr}, *d_edges_{nullptr};

  // CUDA execution objects
  cudaStream_t stream_;
  cudaEvent_t evt_start_, evt_done_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OptimizedImageNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
