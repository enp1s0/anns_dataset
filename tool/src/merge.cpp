#include "tools.hpp"
#include "utils.hpp"
#include <anns_dataset.hpp>
#include <vector>

template <class T>
int merge_core(const std::string output_path,
               const std::vector<std::string> input_path_list) {
  const auto [dataset_size_0, dataset_dim_0] =
      mtk::anns_dataset::load_size_info<T>(input_path_list[0]);
  const auto format =
      mtk::anns_dataset::detect_file_format<T>(input_path_list[0]);
  UNUSED(dataset_size_0);

  mtk::anns_dataset::store_stream<T> ss(output_path, dataset_dim_0, format);
  mtk::anns_tool::print_info(
      mtk::anns_tool::str_format("Output path : %s", output_path.c_str()));

  std::size_t total_dataset_size = 0;
  std::uint32_t num_processed = 0;
  for (const auto &input_path : input_path_list) {
    const auto [dataset_size, dataset_dim] =
        mtk::anns_dataset::load_size_info<T>(input_path_list[0]);
    mtk::anns_tool::print_info(
        mtk::anns_tool::str_format("Merging %s [size=%lu] (%3u / %3lu) ...",
                                   input_path.c_str(), dataset_size,
                                   num_processed + 1, input_path_list.size()),
        false);

    if (dataset_dim != dataset_dim_0) {
      std::printf("\n");
      std::fflush(stdout);
      mtk::anns_tool::print_error(mtk::anns_tool::str_format(
          "Inconsistent dataset dim. [%s].dim = %lu v.s. "
          "[%s].dim = %lu",
          input_path_list[0].c_str(), dataset_dim_0, input_path.c_str(),
          dataset_dim));
      return 1;
    }

    std::vector<T> dataset_buffer(dataset_dim * dataset_size);
    mtk::anns_dataset::load(dataset_buffer.data(), input_path);

    ss.append(dataset_buffer.data(), dataset_dim, dataset_size);

    std::printf("\n");
    std::fflush(stdout);
    num_processed++;
    total_dataset_size += dataset_size;
  }

  mtk::anns_tool::print_info(mtk::anns_tool::str_format(
      "Total dataset size : %lu", total_dataset_size));
  mtk::anns_tool::print_info(
      mtk::anns_tool::str_format("Closing %s", output_path.c_str()));
  ss.close();

  return 0;
}

int mtk::anns_tool::merge(mtk::anns_tool::arguments_t args) {
  mtk::anns_tool::print_info(
      mtk::anns_tool::str_format("Function = %s", __func__));
  const auto params = mtk::anns_tool::parse_params(args.params);

  const auto input_path_list = mtk::anns_tool::split_str(args.input_path, ':');
  mtk::anns_tool::print_info("Input file list");
  for (std::uint32_t i = 0; i < input_path_list.size(); i++) {
    mtk::anns_tool::print_info(
        mtk::anns_tool::str_format("(%3u) %s", i, input_path_list[i].c_str()));
  }
  if (args.dtype == "float") {
    return merge_core<float>(args.output_path, input_path_list);
  } else if (args.dtype == "int8") {
    return merge_core<std::int8_t>(args.output_path, input_path_list);
  } else if (args.dtype == "uint8") {
    return merge_core<std::uint8_t>(args.output_path, input_path_list);
  } else {
    std::fprintf(stderr, "[merge] Invalid data type %s\n", args.dtype.c_str());
    return 1;
  }
  return 0;
}
