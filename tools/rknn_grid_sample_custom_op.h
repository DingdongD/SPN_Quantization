#ifndef SPN_RKNN_GRID_SAMPLE_CUSTOM_OP_H_
#define SPN_RKNN_GRID_SAMPLE_CUSTOM_OP_H_

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>

#include "rknn_api.h"
#include "rknn_custom_op.h"

#include "rknn_grid_sample.h"

namespace spn_rknn {

inline std::atomic<uint64_t> grid_sample_time_ns{0};
inline std::atomic<uint64_t> grid_sample_call_count{0};
inline std::atomic<bool> grid_sample_attributes_printed{false};

inline void reset_grid_sample_stats() {
  grid_sample_time_ns.store(0, std::memory_order_relaxed);
  grid_sample_call_count.store(0, std::memory_order_relaxed);
}

inline uint64_t current_grid_sample_time_ns() {
  return grid_sample_time_ns.load(std::memory_order_relaxed);
}

inline uint64_t current_grid_sample_call_count() {
  return grid_sample_call_count.load(std::memory_order_relaxed);
}

inline void print_tensor(const char* label, const rknn_custom_op_tensor& tensor) {
  std::fprintf(stderr, "%s type=%s fmt=%s dims=", label,
               get_type_string(tensor.attr.type),
               get_format_string(tensor.attr.fmt));
  for (uint32_t index = 0; index < tensor.attr.n_dims; ++index) {
    std::fprintf(stderr, "%s%u", index ? "x" : "", tensor.attr.dims[index]);
  }
  std::fprintf(stderr, " size=%u size_with_stride=%u offset=%d\n",
               tensor.attr.size, tensor.attr.size_with_stride,
               tensor.mem.offset);
}

template <typename Input, typename Grid, typename Output>
inline int compute_grid_sample_typed(
    const rknn_custom_op_tensor& input_tensor,
    const rknn_custom_op_tensor& grid_tensor,
    rknn_custom_op_tensor& output_tensor) {
  if (input_tensor.attr.n_dims != 4 || grid_tensor.attr.n_dims != 4 ||
      output_tensor.attr.n_dims != 4) {
    std::fprintf(stderr, "GridSample requires rank-4 tensors\n");
    return -1;
  }
  const bool input_is_nhwc = input_tensor.attr.fmt == RKNN_TENSOR_NHWC;
  const bool output_is_nhwc = output_tensor.attr.fmt == RKNN_TENSOR_NHWC;
  const auto& in_dims = input_tensor.attr.dims;
  const auto& grid_dims = grid_tensor.attr.dims;
  const auto& out_dims = output_tensor.attr.dims;
  const int64_t batch = in_dims[0];
  const int64_t channels = input_is_nhwc ? in_dims[3] : in_dims[1];
  const int64_t input_height = input_is_nhwc ? in_dims[1] : in_dims[2];
  const int64_t input_width = input_is_nhwc ? in_dims[2] : in_dims[3];
  const bool grid_is_nchw = grid_dims[1] == 2 && grid_dims[3] != 2;
  const int64_t output_height = grid_is_nchw ? grid_dims[2] : grid_dims[1];
  const int64_t output_width = grid_is_nchw ? grid_dims[3] : grid_dims[2];
  const int64_t output_batch = out_dims[0];
  const int64_t output_channels = output_is_nhwc ? out_dims[3] : out_dims[1];
  const int64_t reported_height = output_is_nhwc ? out_dims[1] : out_dims[2];
  const int64_t reported_width = output_is_nhwc ? out_dims[2] : out_dims[3];
  if (grid_dims[0] != batch || output_batch != batch ||
      output_channels != channels || reported_height != output_height ||
      reported_width != output_width) {
    std::fprintf(stderr, "GridSample tensor shapes are inconsistent\n");
    return -1;
  }

  const auto* input = reinterpret_cast<const Input*>(
      static_cast<const unsigned char*>(input_tensor.mem.virt_addr) +
      input_tensor.mem.offset);
  const auto* grid = reinterpret_cast<const Grid*>(
      static_cast<const unsigned char*>(grid_tensor.mem.virt_addr) +
      grid_tensor.mem.offset);
  auto* output = reinterpret_cast<Output*>(
      static_cast<unsigned char*>(output_tensor.mem.virt_addr) +
      output_tensor.mem.offset);
  grid_sample_bilinear_zeros_align_corners_false(
      input, grid, output, batch, channels, input_height, input_width,
      output_height, output_width, input_is_nhwc, grid_is_nchw,
      output_is_nhwc);
  return 0;
}

inline int compute_grid_sample(
    rknn_custom_op_context*, rknn_custom_op_tensor* inputs, uint32_t n_inputs,
    rknn_custom_op_tensor* outputs, uint32_t n_outputs) {
  if (n_inputs != 2 || n_outputs != 1) {
    std::fprintf(stderr, "GridSample expects two inputs and one output\n");
    return -1;
  }
  if (!grid_sample_attributes_printed.exchange(true)) {
    print_tensor("GridSample input", inputs[0]);
    print_tensor("GridSample grid", inputs[1]);
    print_tensor("GridSample output", outputs[0]);
  }
  const auto start = std::chrono::steady_clock::now();
  int status = -1;
  if (inputs[0].attr.type == RKNN_TENSOR_FLOAT16 &&
      inputs[1].attr.type == RKNN_TENSOR_FLOAT16 &&
      outputs[0].attr.type == RKNN_TENSOR_FLOAT16) {
    status = compute_grid_sample_typed<__fp16, __fp16, __fp16>(
        inputs[0], inputs[1], outputs[0]);
  } else if (inputs[0].attr.type == RKNN_TENSOR_FLOAT32 &&
             inputs[1].attr.type == RKNN_TENSOR_FLOAT32 &&
             outputs[0].attr.type == RKNN_TENSOR_FLOAT32) {
    status = compute_grid_sample_typed<float, float, float>(
        inputs[0], inputs[1], outputs[0]);
  } else {
    std::fprintf(stderr, "GridSample supports matching FP16 or FP32 tensors\n");
  }
  const auto end = std::chrono::steady_clock::now();
  grid_sample_time_ns.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count(),
      std::memory_order_relaxed);
  grid_sample_call_count.fetch_add(1, std::memory_order_relaxed);
  return status;
}

inline int register_grid_sample_custom_op(rknn_context context) {
  static rknn_custom_op custom_op{};
  std::memset(&custom_op, 0, sizeof(custom_op));
  std::strncpy(custom_op.op_type, "GridSample", RKNN_MAX_NAME_LEN - 1);
  custom_op.version = 1;
  custom_op.target = RKNN_TARGET_TYPE_CPU;
  custom_op.compute = compute_grid_sample;
  return rknn_register_custom_ops(context, &custom_op, 1);
}

}  // namespace spn_rknn

#endif  // SPN_RKNN_GRID_SAMPLE_CUSTOM_OP_H_
