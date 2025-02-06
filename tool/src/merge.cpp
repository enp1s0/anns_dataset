#include <anns_dataset.hpp>
#include <vector>

#define UNUSED(a)                                                              \
  do {                                                                         \
    (void)(a);                                                                 \
  } while (0)

template <class T>
int merge_core(const std::string output_path,
               const std::vector<std::string> input_path_list) {
  const auto [dataset_size_0, dataset_dim_0] =
      mtk::anns_dataset::load_size_info<T>(input_path_list[0]);
  const auto format =
      mtk::anns_dataset::detect_file_format<T>(input_path_list[0]);
  UNUSED(dataset_size_0);

  mtk::anns_dataset::store_stream<T> ss(output_path, dataset_dim_0, format);

  std::uint32_t num_processed = 0;
  for (const auto &input_path : input_path_list) {
    std::printf("[merge] Merging %s (%3u / %3lu) ...", input_path.c_str(),
                num_processed + 1, input_path_list.size());
    const auto [dataset_size, dataset_dim] =
        mtk::anns_dataset::load_size_info<T>(input_path_list[0]);
    if (dataset_dim != dataset_dim_0) {
      std::printf("\n");
      std::fprintf(stderr,
                   "[merge] Inconsistent dataset dim. [%s].dim = %lu v.s. "
                   "[%s].dim = %lu\n",
                   input_path_list[0].c_str(), dataset_dim_0,
                   input_path.c_str(), dataset_dim);
      return 1;
    }

    std::vector<T> dataset_buffer(dataset_dim * dataset_size);
    mtk::anns_dataset::load(dataset_buffer.data(), input_path);

    ss.append(dataset_buffer.data(), dataset_dim, dataset_size);

    std::printf(" Done\n");
    num_processed++;
  }

  ss.close();

  return 0;
}

int main(int argc, char **argv) {
  if (argc <= 3) {
    std::fprintf(stderr,
                 "Usage: %s [dtype (int8, uint8, float)] [output_path] "
                 "[input_path 0] [input_path 1] ...\n",
                 argv[0]);
    return 1;
  }

  const std::string dtype(argv[1]);
  const std::string output_path(argv[2]);
  std::vector<std::string> input_path_list;
  for (std::uint32_t i = 3; i < static_cast<std::uint32_t>(argc); i++) {
    input_path_list.push_back(std::string(argv[i]));
  }

  if (dtype == "float") {
    return merge_core<float>(output_path, input_path_list);
  } else if (dtype == "int8") {
    return merge_core<std::int8_t>(output_path, input_path_list);
  } else if (dtype == "uint8") {
    return merge_core<std::uint8_t>(output_path, input_path_list);
  } else {
    std::fprintf(stderr, "[merge] Invalid data type %s\n", dtype.c_str());
    return 1;
  }
  return 0;
}
