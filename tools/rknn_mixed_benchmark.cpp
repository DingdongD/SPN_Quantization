#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
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

std::string quote(const std::string& value) {
  std::string output = "\"";
  for (char character : value) {
    if (character == '\\' || character == '"') output += '\\';
    output += character;
  }
  output += '"';
  return output;
}

template <typename Value>
void write_array(std::ostream& stream, const std::vector<Value>& values) {
  stream << '[';
  for (size_t index = 0; index < values.size(); ++index) {
    if (index) stream << ',';
    stream << values[index];
  }
  stream << ']';
}

struct OutputStats {
  double minimum = 0.0;
  double maximum = 0.0;
  double mean = 0.0;
  uint32_t finite_count = 0;
  uint32_t nan_count = 0;
  uint32_t positive_inf_count = 0;
  uint32_t negative_inf_count = 0;
};

OutputStats summarize_output(const rknn_output& output, uint32_t elements) {
  if (output.buf == nullptr || elements == 0) return {};
  const auto* values = static_cast<const float*>(output.buf);
  OutputStats stats;
  stats.minimum = std::numeric_limits<double>::infinity();
  stats.maximum = -std::numeric_limits<double>::infinity();
  long double sum = 0.0;
  for (uint32_t index = 0; index < elements; ++index) {
    const double value = values[index];
    if (std::isnan(value)) {
      ++stats.nan_count;
    } else if (std::isinf(value)) {
      if (value > 0) {
        ++stats.positive_inf_count;
      } else {
        ++stats.negative_inf_count;
      }
    } else {
      stats.minimum = std::min(stats.minimum, value);
      stats.maximum = std::max(stats.maximum, value);
      sum += value;
      ++stats.finite_count;
    }
  }
  if (stats.finite_count == 0) {
    stats.minimum = 0.0;
    stats.maximum = 0.0;
  } else {
    stats.mean = static_cast<double>(sum / stats.finite_count);
  }
  return stats;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 8) {
    std::cerr << "usage: rknn_mixed_benchmark MODEL_NAME MODEL.rknn "
                 "OUTPUT.json WARMUPS TIMED_PER_ROUND ROUNDS "
                 "INPUT0.f32 [INPUT1.f32 ...]\n";
    return 2;
  }
  const std::string model_name = argv[1];
  const std::string model_path = argv[2];
  const std::string output_path = argv[3];
  const int warmups = std::stoi(argv[4]);
  const int timed_per_round = std::stoi(argv[5]);
  const int rounds = std::stoi(argv[6]);
  if (warmups < 1 || timed_per_round < 1 || rounds < 1) {
    std::cerr << "iteration counts must be positive\n";
    return 2;
  }

  rknn_context context = 0;
  try {
    check(rknn_init(&context, const_cast<char*>(model_path.c_str()), 0, 0,
                    nullptr), "rknn_init");
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
    if (static_cast<int>(counts.n_input) != argc - 7) {
      throw std::runtime_error("input file count does not match model");
    }

    std::vector<std::vector<unsigned char>> buffers;
    std::vector<rknn_input> inputs(counts.n_input);
    std::vector<std::string> input_paths;
    for (uint32_t index = 0; index < counts.n_input; ++index) {
      rknn_tensor_attr attribute{};
      attribute.index = index;
      check(rknn_query(context, RKNN_QUERY_INPUT_ATTR, &attribute,
                       sizeof(attribute)), "RKNN_QUERY_INPUT_ATTR");
      input_paths.emplace_back(argv[index + 7]);
      buffers.push_back(read_binary(input_paths.back()));
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

    rknn_tensor_attr output_attribute{};
    output_attribute.index = 0;
    check(rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &output_attribute,
                     sizeof(output_attribute)), "RKNN_QUERY_OUTPUT_ATTR");

    auto run_once = [&](bool save_output, OutputStats* stats,
                        std::vector<float>* saved_output) {
      std::vector<rknn_output> outputs(counts.n_output);
      for (uint32_t index = 0; index < counts.n_output; ++index) {
        outputs[index].index = index;
        outputs[index].want_float = 1;
        outputs[index].is_prealloc = 0;
      }
      spn_rknn::reset_grid_sample_stats();
      const auto start = std::chrono::steady_clock::now();
      check(rknn_run(context, nullptr), "rknn_run");
      check(rknn_outputs_get(context, counts.n_output, outputs.data(), nullptr),
            "rknn_outputs_get");
      const auto end = std::chrono::steady_clock::now();
      const double latency_ms =
          std::chrono::duration<double, std::milli>(end - start).count();
      const double grid_sample_ms =
          spn_rknn::current_grid_sample_time_ns() / 1.0e6;
      const uint64_t grid_sample_calls =
          spn_rknn::current_grid_sample_call_count();
      if (save_output && stats != nullptr) {
        *stats = summarize_output(outputs[0], output_attribute.n_elems);
        const auto* values = static_cast<const float*>(outputs[0].buf);
        saved_output->assign(values, values + output_attribute.n_elems);
      }
      check(rknn_outputs_release(context, counts.n_output, outputs.data()),
            "rknn_outputs_release");
      return std::make_tuple(latency_ms, grid_sample_ms, grid_sample_calls);
    };

    for (int iteration = 0; iteration < warmups; ++iteration) {
      run_once(false, nullptr, nullptr);
    }
    std::vector<double> latency_ms;
    std::vector<double> grid_sample_cpu_ms;
    std::vector<uint64_t> grid_sample_calls;
    OutputStats output_stats;
    std::vector<float> final_output;
    for (int round = 0; round < rounds; ++round) {
      for (int iteration = 0; iteration < timed_per_round; ++iteration) {
        const bool final_iteration =
            round == rounds - 1 && iteration == timed_per_round - 1;
        auto [latency, grid_time, grid_calls] =
            run_once(final_iteration, &output_stats, &final_output);
        latency_ms.push_back(latency);
        grid_sample_cpu_ms.push_back(grid_time);
        grid_sample_calls.push_back(grid_calls);
      }
    }

    rknn_mem_size memory{};
    check(rknn_query(context, RKNN_QUERY_MEM_SIZE, &memory, sizeof(memory)),
          "RKNN_QUERY_MEM_SIZE");
    const std::string output_dump_path = output_path + ".output0.f32";
    std::ofstream output_dump(output_dump_path, std::ios::binary);
    if (!output_dump) {
      throw std::runtime_error("cannot create " + output_dump_path);
    }
    output_dump.write(
        reinterpret_cast<const char*>(final_output.data()),
        static_cast<std::streamsize>(final_output.size() * sizeof(float)));
    if (!output_dump) {
      throw std::runtime_error("cannot write " + output_dump_path);
    }
    output_dump.close();
    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("cannot create " + output_path);
    output << std::setprecision(12);
    output << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"model\": " << quote(model_name) << ",\n"
           << "  \"model_path\": " << quote(model_path) << ",\n"
           << "  \"precision\": \"fp16\",\n"
           << "  \"measurement_scope\": \"synchronous rknn_run plus "
              "rknn_outputs_get; excludes model loading, input setup, output "
              "statistics, and rknn_outputs_release\",\n"
           << "  \"api_version\": " << quote(versions.api_version) << ",\n"
           << "  \"driver_version\": " << quote(versions.drv_version) << ",\n"
           << "  \"warmup_iterations\": " << warmups << ",\n"
           << "  \"timed_iterations_per_round\": " << timed_per_round << ",\n"
           << "  \"rounds\": " << rounds << ",\n"
           << "  \"core_mask\": \"AUTO\",\n"
           << "  \"input_paths\": [";
    for (size_t index = 0; index < input_paths.size(); ++index) {
      if (index) output << ',';
      output << quote(input_paths[index]);
    }
    output << "],\n  \"output_dump_path\": " << quote(output_dump_path)
           << ",\n  \"raw_latency_ms\": ";
    write_array(output, latency_ms);
    output << ",\n  \"raw_grid_sample_cpu_ms\": ";
    write_array(output, grid_sample_cpu_ms);
    output << ",\n  \"raw_grid_sample_call_count\": ";
    write_array(output, grid_sample_calls);
    output << ",\n"
           << "  \"output\": {\"elements\": " << output_attribute.n_elems
           << ", \"all_finite\": "
           << (output_stats.finite_count == output_attribute.n_elems
                   ? "true" : "false")
           << ", \"finite_count\": " << output_stats.finite_count
           << ", \"nan_count\": " << output_stats.nan_count
           << ", \"positive_inf_count\": "
           << output_stats.positive_inf_count
           << ", \"negative_inf_count\": "
           << output_stats.negative_inf_count
           << ", \"finite_minimum\": " << output_stats.minimum
           << ", \"finite_maximum\": " << output_stats.maximum
           << ", \"finite_mean\": " << output_stats.mean << "},\n"
           << "  \"memory_bytes\": {\"weights\": "
           << memory.total_weight_size << ", \"internal\": "
           << memory.total_internal_size << ", \"dma_allocated\": "
           << memory.total_dma_allocated_size << "}\n"
           << "}\n";
    output.close();
    check(rknn_destroy(context), "rknn_destroy");
    context = 0;
    std::cout << "wrote " << output_path << " with " << latency_ms.size()
              << " measured forwards\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    if (context != 0) rknn_destroy(context);
    return 1;
  }
}
