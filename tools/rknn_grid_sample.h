#ifndef SPN_RKNN_GRID_SAMPLE_H_
#define SPN_RKNN_GRID_SAMPLE_H_

#include <cmath>
#include <cstdint>

namespace spn_rknn {

template <typename Input, typename Grid, typename Output>
void grid_sample_bilinear_zeros_align_corners_false(
    const Input* input, const Grid* grid, Output* output,
    int64_t batch, int64_t channels, int64_t input_height,
    int64_t input_width, int64_t output_height, int64_t output_width,
    bool input_is_nhwc, bool grid_is_nchw, bool output_is_nhwc) {
  const auto input_value = [&](int64_t n, int64_t c, int64_t y,
                               int64_t x) -> float {
    if (x < 0 || x >= input_width || y < 0 || y >= input_height) {
      return 0.0f;
    }
    const int64_t index = input_is_nhwc
        ? ((n * input_height + y) * input_width + x) * channels + c
        : ((n * channels + c) * input_height + y) * input_width + x;
    return static_cast<float>(input[index]);
  };

#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t oy = 0; oy < output_height; ++oy) {
        for (int64_t ox = 0; ox < output_width; ++ox) {
          int64_t gx_index;
          int64_t gy_index;
          if (grid_is_nchw) {
            gx_index = ((n * 2) * output_height + oy) * output_width + ox;
            gy_index = ((n * 2 + 1) * output_height + oy) * output_width + ox;
          } else {
            gx_index = ((n * output_height + oy) * output_width + ox) * 2;
            gy_index = gx_index + 1;
          }
          const float gx = static_cast<float>(grid[gx_index]);
          const float gy = static_cast<float>(grid[gy_index]);
          const float ix = ((gx + 1.0f) * input_width - 1.0f) * 0.5f;
          const float iy = ((gy + 1.0f) * input_height - 1.0f) * 0.5f;
          const int64_t x0 = static_cast<int64_t>(std::floor(ix));
          const int64_t y0 = static_cast<int64_t>(std::floor(iy));
          const int64_t x1 = x0 + 1;
          const int64_t y1 = y0 + 1;
          const float wx1 = ix - static_cast<float>(x0);
          const float wy1 = iy - static_cast<float>(y0);
          const float wx0 = 1.0f - wx1;
          const float wy0 = 1.0f - wy1;
          const float value =
              input_value(n, c, y0, x0) * wx0 * wy0 +
              input_value(n, c, y0, x1) * wx1 * wy0 +
              input_value(n, c, y1, x0) * wx0 * wy1 +
              input_value(n, c, y1, x1) * wx1 * wy1;
          const int64_t output_index = output_is_nhwc
              ? ((n * output_height + oy) * output_width + ox) * channels + c
              : ((n * channels + c) * output_height + oy) * output_width + ox;
          output[output_index] = static_cast<Output>(value);
        }
      }
    }
  }
}

template <typename Input, typename Grid, typename Output>
void grid_sample_bilinear_zeros_align_corners_false_nchw(
    const Input* input, const Grid* grid, Output* output,
    int64_t batch, int64_t channels, int64_t input_height,
    int64_t input_width, int64_t output_height, int64_t output_width,
    bool grid_is_nchw) {
  grid_sample_bilinear_zeros_align_corners_false(
      input, grid, output, batch, channels, input_height, input_width,
      output_height, output_width, false, grid_is_nchw, false);
}

}  // namespace spn_rknn

#endif  // SPN_RKNN_GRID_SAMPLE_H_
