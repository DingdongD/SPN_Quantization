#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "rknn_api.h"
#include "rknn_grid_sample_custom_op.h"

namespace {

void check(int status, const char* operation) {
  if (status != RKNN_SUCC) {
    throw std::runtime_error(std::string(operation) + " failed: " +
                             std::to_string(status));
  }
}

std::vector<unsigned char> read_binary(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("cannot open " + path);
  const auto size = stream.tellg();
  if (size < 0) throw std::runtime_error("cannot size " + path);
  stream.seekg(0);
  std::vector<unsigned char> bytes(static_cast<size_t>(size));
  stream.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!stream) throw std::runtime_error("cannot read " + path);
  return bytes;
}

std::vector<std::string> split_tabs(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, '\t')) fields.push_back(field);
  return fields;
}

std::string quote(const std::string& value) {
  std::ostringstream stream;
  stream << '"';
  for (char character : value) {
    if (character == '"' || character == '\\') stream << '\\';
    stream << character;
  }
  stream << '"';
  return stream.str();
}

struct SampleMetric {
  int index = -1;
  uint64_t valid_pixels = 0;
  uint64_t nonfinite_prediction_pixels = 0;
  uint64_t nonpositive_prediction_pixels = 0;
  double squared_error_sum = 0.0;
  double absolute_error_sum = 0.0;
  double absolute_relative_error_sum = 0.0;
  double rmse = 0.0;
  double latency_ms = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage: rknn_fullval_evaluator MODEL_NAME MODEL.rknn "
                 "MANIFEST.tsv OUTPUT.json\n";
    return 2;
  }
  const std::string model_name = argv[1];
  const std::string model_path = argv[2];
  const std::string manifest_path = argv[3];
  const std::string output_path = argv[4];
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
    if (counts.n_output < 1 || counts.n_input < 1) {
      throw std::runtime_error("model has no deployment input or output");
    }
    std::vector<rknn_tensor_attr> input_attrs(counts.n_input);
    for (uint32_t i = 0; i < counts.n_input; ++i) {
      input_attrs[i].index = i;
      check(rknn_query(context, RKNN_QUERY_INPUT_ATTR, &input_attrs[i],
                       sizeof(input_attrs[i])), "RKNN_QUERY_INPUT_ATTR");
    }
    rknn_tensor_attr output_attr{};
    output_attr.index = 0;
    check(rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &output_attr,
                     sizeof(output_attr)), "RKNN_QUERY_OUTPUT_ATTR");

    std::ifstream manifest(manifest_path);
    if (!manifest) throw std::runtime_error("cannot open manifest");
    std::vector<SampleMetric> rows;
    std::string line;
    while (std::getline(manifest, line)) {
      if (line.empty()) continue;
      const auto fields = split_tabs(line);
      if (fields.size() != static_cast<size_t>(counts.n_input) + 2) {
        throw std::runtime_error("manifest column count differs from model");
      }
      SampleMetric row;
      row.index = std::stoi(fields[0]);
      std::vector<std::vector<unsigned char>> buffers;
      std::vector<rknn_input> inputs(counts.n_input);
      for (uint32_t i = 0; i < counts.n_input; ++i) {
        buffers.push_back(read_binary(fields[i + 1]));
        const size_t expected =
            static_cast<size_t>(input_attrs[i].n_elems) * sizeof(float);
        if (buffers.back().size() != expected) {
          throw std::runtime_error("input byte count differs from model");
        }
        inputs[i].index = i;
        inputs[i].buf = buffers.back().data();
        inputs[i].size = static_cast<uint32_t>(buffers.back().size());
        inputs[i].pass_through = 0;
        inputs[i].type = RKNN_TENSOR_FLOAT32;
        inputs[i].fmt = RKNN_TENSOR_NCHW;
      }
      const auto truth_bytes = read_binary(fields.back());
      if (truth_bytes.size() !=
          static_cast<size_t>(output_attr.n_elems) * sizeof(float)) {
        throw std::runtime_error("ground-truth byte count differs from output");
      }
      const auto* truth =
          reinterpret_cast<const float*>(truth_bytes.data());
      check(rknn_inputs_set(context, counts.n_input, inputs.data()),
            "rknn_inputs_set");
      std::vector<rknn_output> outputs(counts.n_output);
      for (uint32_t i = 0; i < counts.n_output; ++i) {
        outputs[i].index = i;
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 0;
      }
      const auto start = std::chrono::steady_clock::now();
      check(rknn_run(context, nullptr), "rknn_run");
      check(rknn_outputs_get(context, counts.n_output, outputs.data(), nullptr),
            "rknn_outputs_get");
      const auto end = std::chrono::steady_clock::now();
      row.latency_ms = std::chrono::duration<double, std::milli>(
          end - start).count();
      const auto* prediction = static_cast<const float*>(outputs[0].buf);
      for (uint32_t i = 0; i < output_attr.n_elems; ++i) {
        if (!std::isfinite(truth[i]) || truth[i] <= 1.0e-4f) continue;
        ++row.valid_pixels;
        if (!std::isfinite(prediction[i])) {
          ++row.nonfinite_prediction_pixels;
          continue;
        }
        if (prediction[i] <= 1.0e-4f) {
          ++row.nonpositive_prediction_pixels;
        }
        const double difference =
            static_cast<double>(prediction[i]) - truth[i];
        row.squared_error_sum += difference * difference;
        row.absolute_error_sum += std::abs(difference);
        row.absolute_relative_error_sum +=
            std::abs(difference) / static_cast<double>(truth[i]);
      }
      check(rknn_outputs_release(context, counts.n_output, outputs.data()),
            "rknn_outputs_release");
      if (row.valid_pixels == 0 || row.nonfinite_prediction_pixels != 0) {
        throw std::runtime_error(
            "sample " + std::to_string(row.index) +
            " has invalid task-evaluation output: valid_pixels=" +
            std::to_string(row.valid_pixels) +
            " nonfinite_prediction_pixels=" +
            std::to_string(row.nonfinite_prediction_pixels));
      }
      row.rmse = std::sqrt(
          row.squared_error_sum / static_cast<double>(row.valid_pixels));
      rows.push_back(row);
      if (rows.size() % 25 == 0) {
        std::cerr << model_name << " samples=" << rows.size() << '\n';
      }
    }
    if (rows.size() != 654) {
      throw std::runtime_error("full validation requires exactly 654 samples");
    }
    double squared_error_sum = 0.0;
    double absolute_error_sum = 0.0;
    double absolute_relative_error_sum = 0.0;
    double sample_rmse_sum = 0.0;
    double latency_sum = 0.0;
    uint64_t valid_pixels = 0;
    uint64_t nonfinite = 0;
    uint64_t nonpositive = 0;
    for (const auto& row : rows) {
      squared_error_sum += row.squared_error_sum;
      absolute_error_sum += row.absolute_error_sum;
      absolute_relative_error_sum += row.absolute_relative_error_sum;
      sample_rmse_sum += row.rmse;
      latency_sum += row.latency_ms;
      valid_pixels += row.valid_pixels;
      nonfinite += row.nonfinite_prediction_pixels;
      nonpositive += row.nonpositive_prediction_pixels;
    }
    std::ofstream output(output_path);
    if (!output) throw std::runtime_error("cannot create output JSON");
    output << std::setprecision(12)
           << "{\n  \"schema_version\": 1,\n"
           << "  \"model\": " << quote(model_name) << ",\n"
           << "  \"model_path\": " << quote(model_path) << ",\n"
           << "  \"manifest_path\": " << quote(manifest_path) << ",\n"
           << "  \"api_version\": " << quote(versions.api_version) << ",\n"
           << "  \"driver_version\": " << quote(versions.drv_version) << ",\n"
           << "  \"sample_count\": " << rows.size() << ",\n"
           << "  \"valid_pixels\": " << valid_pixels << ",\n"
           << "  \"nonfinite_prediction_pixels\": " << nonfinite << ",\n"
           << "  \"nonpositive_prediction_pixels\": " << nonpositive << ",\n"
           << "  \"mean_sample_rmse_m\": "
           << sample_rmse_sum / rows.size() << ",\n"
           << "  \"pooled_rmse_m\": "
           << std::sqrt(squared_error_sum / valid_pixels) << ",\n"
           << "  \"mae_m\": " << absolute_error_sum / valid_pixels << ",\n"
           << "  \"abs_rel\": "
           << absolute_relative_error_sum / valid_pixels << ",\n"
           << "  \"mean_forward_latency_ms\": "
           << latency_sum / rows.size() << ",\n"
           << "  \"samples\": [\n";
    for (size_t i = 0; i < rows.size(); ++i) {
      const auto& row = rows[i];
      output << "    {\"index\": " << row.index
             << ", \"valid_pixels\": " << row.valid_pixels
             << ", \"rmse_m\": " << row.rmse
             << ", \"latency_ms\": " << row.latency_ms << "}"
             << (i + 1 == rows.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    check(rknn_destroy(context), "rknn_destroy");
    context = 0;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    if (context) rknn_destroy(context);
    return 1;
  }
}
