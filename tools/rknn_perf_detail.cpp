#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rknn_api.h"
#include "rknn_grid_sample_custom_op.h"

namespace {

std::vector<unsigned char> read_binary(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("cannot open " + path);
  const auto end = stream.tellg();
  if (end < 0) throw std::runtime_error("cannot size " + path);
  std::vector<unsigned char> bytes(static_cast<size_t>(end));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(bytes.data()), end);
  if (!stream) throw std::runtime_error("cannot read " + path);
  return bytes;
}

void check(int status, const char* operation) {
  if (status != RKNN_SUCC) {
    throw std::runtime_error(
        std::string(operation) + " failed with " + std::to_string(status));
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "usage: rknn_perf_detail MODEL.rknn INPUT0.f32 [INPUT1.f32 ...]\n";
    return 2;
  }
  rknn_context context = 0;
  try {
    check(rknn_init(&context, argv[1], 0, RKNN_FLAG_COLLECT_PERF_MASK, nullptr),
          "rknn_init");
    check(spn_rknn::register_grid_sample_custom_op(context),
          "rknn_register_custom_ops(GridSample)");
    check(rknn_set_core_mask(context, RKNN_NPU_CORE_AUTO),
          "rknn_set_core_mask");

    rknn_sdk_version versions{};
    check(rknn_query(context, RKNN_QUERY_SDK_VERSION, &versions,
                     sizeof(versions)), "RKNN_QUERY_SDK_VERSION");
    rknn_input_output_num counts{};
    check(rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &counts, sizeof(counts)),
          "RKNN_QUERY_IN_OUT_NUM");
    if (static_cast<int>(counts.n_input) != argc - 2) {
      throw std::runtime_error("input file count does not match model");
    }

    std::vector<std::vector<unsigned char>> buffers;
    std::vector<rknn_input> inputs(counts.n_input);
    for (uint32_t index = 0; index < counts.n_input; ++index) {
      rknn_tensor_attr attribute{};
      attribute.index = index;
      check(rknn_query(context, RKNN_QUERY_INPUT_ATTR, &attribute,
                       sizeof(attribute)), "RKNN_QUERY_INPUT_ATTR");
      buffers.push_back(read_binary(argv[index + 2]));
      const size_t expected_size = static_cast<size_t>(attribute.n_elems) * 4;
      if (buffers.back().size() != expected_size) {
        throw std::runtime_error(
            "input " + std::to_string(index) + " has " +
            std::to_string(buffers.back().size()) + " bytes, expected " +
            std::to_string(expected_size));
      }
      inputs[index].index = index;
      inputs[index].buf = buffers.back().data();
      inputs[index].size = static_cast<uint32_t>(buffers.back().size());
      inputs[index].pass_through = 0;
      inputs[index].type = RKNN_TENSOR_FLOAT32;
      inputs[index].fmt = RKNN_TENSOR_NCHW;
    }
    check(rknn_inputs_set(context, counts.n_input, inputs.data()),
          "rknn_inputs_set");
    check(rknn_run(context, nullptr), "rknn_run");

    std::vector<rknn_output> outputs(counts.n_output);
    for (uint32_t index = 0; index < counts.n_output; ++index) {
      outputs[index].index = index;
      outputs[index].want_float = 1;
      outputs[index].is_prealloc = 0;
    }
    check(rknn_outputs_get(context, counts.n_output, outputs.data(), nullptr),
          "rknn_outputs_get");

    rknn_perf_run run{};
    check(rknn_query(context, RKNN_QUERY_PERF_RUN, &run, sizeof(run)),
          "RKNN_QUERY_PERF_RUN");
    rknn_perf_detail detail{};
    check(rknn_query(context, RKNN_QUERY_PERF_DETAIL, &detail, sizeof(detail)),
          "RKNN_QUERY_PERF_DETAIL");
    rknn_mem_size memory{};
    check(rknn_query(context, RKNN_QUERY_MEM_SIZE, &memory, sizeof(memory)),
          "RKNN_QUERY_MEM_SIZE");

    std::cout << "api_version=" << versions.api_version << "\n";
    std::cout << "driver_version=" << versions.drv_version << "\n";
    std::cout << "run_duration_us=" << run.run_duration << "\n";
    std::cout << "weight_bytes=" << memory.total_weight_size << "\n";
    std::cout << "internal_bytes=" << memory.total_internal_size << "\n";
    std::cout << "dma_allocated_bytes=" << memory.total_dma_allocated_size
              << "\n";
    std::cout << "perf_detail_begin\n";
    if (detail.perf_data != nullptr && detail.data_len > 0) {
      std::cout.write(detail.perf_data, detail.data_len);
      if (detail.perf_data[detail.data_len - 1] != '\n') std::cout << '\n';
    }
    std::cout << "perf_detail_end\n";

    check(rknn_outputs_release(context, counts.n_output, outputs.data()),
          "rknn_outputs_release");
    check(rknn_destroy(context), "rknn_destroy");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    if (context != 0) rknn_destroy(context);
    return 1;
  }
}
