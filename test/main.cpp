#include <anns_dataset.hpp>
#include <statistic.hpp>

#include <cstdint>
#include <iostream>
#include <vector>

namespace {
template <class T> const std::string to_str();
template <> const std::string to_str<float>() { return "F32"; }
template <> const std::string to_str<std::uint32_t>() { return "U32"; }
template <> const std::string to_str<std::uint64_t>() { return "U64"; }
template <> const std::string to_str<std::int8_t>() { return "I8"; }
template <> const std::string to_str<std::uint8_t>() { return "U8"; }
} // unnamed namespace

std::uint32_t num_processed_test = 0;
std::uint32_t num_passed_test = 0;
std::vector<std::string> failed_test_list;

int check_expected_true(const bool v, const std::string test_name,
                        const std::string case_name) {
  std::printf("[TEST %u] >> %s (%s)\n", num_processed_test, case_name.c_str(),
              test_name.c_str());
  if (v) {
    std::printf("[TEST %u] << PASSED\n", num_processed_test);
    num_passed_test++;
  } else {
    std::printf("[TEST %u] << FAILED\n", num_processed_test);
    failed_test_list.push_back("[" + std::to_string(num_processed_test) + "] " +
                               test_name + " (" + case_name + ")");
  }
  num_processed_test++;
  return !v;
}

#define EXPECTED_TRUE(v, c, t)                                                 \
  if (check_expected_true((v), c, t)) {                                        \
    return;                                                                    \
  }

template <class data_t, class index_t>
void test_core(const std::size_t dataset_size, const std::uint32_t dataset_dim,
               const mtk::anns_dataset::format_t file_format) {
  const std::string test_name =
      "Shape=" + std::to_string(dataset_dim) + "x" +
      std::to_string(dataset_size) + ", DataT=" + to_str<data_t>() +
      ", IdxT=" + to_str<index_t>() +
      ", Fmt=" + mtk::anns_dataset::get_format_str(file_format);
  const std::string file_name = "dataset.dat";

  // Generate a test dataset
  const auto src_dataset_ld = dataset_dim;
  std::vector<data_t> src_dataset(dataset_size * src_dataset_ld);
  for (std::size_t i = 0; i < dataset_size; i++) {
    for (std::uint32_t j = 0; j < dataset_dim; j++) {
      src_dataset[i * src_dataset_ld + j] = ((i + j + 1) * (i + j + 1)) % 128;
    }
  }

  // Store
  mtk::anns_dataset::store(file_name, dataset_size, dataset_dim,
                           src_dataset.data(), file_format);

  // Entire load test
  {
    const auto [dataset_size_load, dataset_dim_load] =
        mtk::anns_dataset::load_size_info<data_t>(file_name);

    // check size
    EXPECTED_TRUE(dataset_size_load == dataset_size, test_name,
                  "Check dataset size of loaded dataset");
    EXPECTED_TRUE(dataset_dim == dataset_dim_load, test_name,
                  "Check dataset dim of loaded dataset");

    const auto dataset_ld = dataset_dim;
    std::vector<data_t> dataset(dataset_size * dataset_ld);
    mtk::anns_dataset::load(dataset.data(), file_name);

    // check data
    bool error = false;
    for (std::size_t i = 0; i < dataset_size; i++) {
      for (std::uint32_t j = 0; j < dataset_dim; j++) {
        error = error || (dataset[i * dataset_ld + j] !=
                          src_dataset[i * src_dataset_ld + j]);
      }
    }
    EXPECTED_TRUE(!error, test_name, "Check dataset data");
  }

  // Partial load test
  {
    const std::size_t offset = dataset_size / 10;
    const std::size_t size = dataset_size / 10;

    const auto dataset_ld = dataset_dim;
    std::vector<data_t> dataset(size * dataset_ld);
    mtk::anns_dataset::load(
        dataset.data(), file_name, false,
        mtk::anns_dataset::format_t::FORMAT_AUTO_DETECT,
        mtk::anns_dataset::range_t{.offset = offset, .size = size});

    // check data
    bool error = false;
    for (std::size_t i = 0; i < size; i++) {
      for (std::uint32_t j = 0; j < dataset_dim; j++) {
        error = error || (dataset[i * dataset_ld + j] !=
                          src_dataset[(offset + i) * src_dataset_ld + j]);
      }
    }
    EXPECTED_TRUE(!error, test_name, "Check partial load dataset data");
  }

  // Store stream
  {
    mtk::anns_dataset::store_stream<data_t> ss(file_name, dataset_dim,
                                               file_format);
    const std::size_t num_split = 10;
    for (std::size_t i = 0; i < num_split; i++) {
      const auto offset = i * dataset_size / num_split;
      const auto size = (i + 1) * dataset_size / num_split - offset;

      ss.append(src_dataset.data() + offset * src_dataset_ld, src_dataset_ld,
                size);
    }
    ss.close();

    const auto dataset_ld = dataset_dim;
    std::vector<data_t> dataset(dataset_size * dataset_ld);
    mtk::anns_dataset::load(dataset.data(), file_name, false,
                            mtk::anns_dataset::format_t::FORMAT_AUTO_DETECT);

    // check data
    bool error = false;
    for (std::size_t i = 0; i < dataset_size; i++) {
      for (std::uint32_t j = 0; j < dataset_dim; j++) {
        error = error || (dataset[i * dataset_ld + j] !=
                          src_dataset[i * src_dataset_ld + j]);
      }
    }
    EXPECTED_TRUE(!error, test_name, "Check store stream");
  }
}

template <class data_t, class index_t> void test() {
  for (const auto &format : std::vector<mtk::anns_dataset::format_t>{
           mtk::anns_dataset::format_t::FORMAT_BIGANN,
           mtk::anns_dataset::format_t::FORMAT_VECS}) {
    for (const auto &dataset_shape :
         std::vector<std::pair<std::uint32_t, std::size_t>>{
             {15u, 1000lu}, {32u, 10000lu}, {1011u, 1000lu}}) {
      test_core<data_t, index_t>(std::get<1>(dataset_shape),
                                 std::get<0>(dataset_shape), format);
    }
  }
}

template <class data_t>
void stats_test_core(const std::size_t dataset_size,
                     const std::size_t dataset_dim) {
  const std::size_t dataset_ld = dataset_dim + 1;
  std::vector<data_t> dataset(dataset_size * dataset_ld);
  for (std::size_t i = 0; i < dataset_size; i++) {
    for (std::size_t j = 0; j < dataset_dim; j++) {
      dataset[i * dataset_ld + j] = (i * 13 + j + 11) % 17;
    }
  }
  mtk::anns_dataset::print_dimensionwise_distribution(
      dataset.data(), dataset_ld, dataset_size, dataset_dim, 40);
  num_passed_test++;
  num_processed_test++;
}

template <class data_t> void stats_test() {
  for (const auto &dataset_shape :
       std::vector<std::pair<std::uint32_t, std::size_t>>{{15u, 1000lu},
                                                          {32u, 10000lu}}) {
    stats_test_core<data_t>(std::get<1>(dataset_shape),
                            std::get<0>(dataset_shape));
  }
}

int main() {
  test<float, std::uint32_t>();
  test<float, std::uint64_t>();
  test<std::uint8_t, std::uint32_t>();
  test<std::uint8_t, std::uint64_t>();
  test<std::int8_t, std::uint32_t>();
  test<std::int8_t, std::uint64_t>();
  stats_test<float>();
  stats_test<std::int8_t>();
  stats_test<std::uint8_t>();
  std::printf("%5u / %5u PASSED\n", num_passed_test, num_processed_test);
  if (!failed_test_list.empty()) {
    std::printf("FAILED TEST(S)\n");
    for (const auto &l : failed_test_list) {
      std::cout << l << std::endl;
    }
  }

  return !(num_processed_test == num_passed_test);
}
