#pragma once
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <omp.h>
#include <vector>

namespace mtk::anns_dataset {
namespace detail {
template <class T> struct stat {
  T min = std::numeric_limits<T>::max(), max = -std::numeric_limits<T>::max();
  double avg = 0;
  double var = 0;
};
} // namespace detail
template <class T>
inline void print_dimensionwise_distribution(
    const T *const dataset_ptr, const std::size_t dataset_ld,
    const std::size_t dataset_size, const std::size_t dataset_dim,
    const std::uint32_t graph_width = 0) {
  std::vector<detail::stat<T>> stats(dataset_dim);

#pragma omp parallel
  {
    std::vector<detail::stat<T>> local_stats(dataset_dim);
    for (std::size_t i = omp_get_thread_num(); i < dataset_size;
         i += omp_get_num_threads()) {
      for (std::size_t j = 0; j < dataset_dim; j++) {
        const auto v = dataset_ptr[j + i * dataset_ld];
        local_stats[j].max = std::max(local_stats[j].max, v);
        local_stats[j].min = std::min(local_stats[j].min, v);
        local_stats[j].avg += v;
      }
    }
#pragma omp critical
    {
      for (std::size_t j = 0; j < dataset_dim; j++) {
        stats[j].max = std::max(stats[j].max, local_stats[j].max);
        stats[j].min = std::min(stats[j].min, local_stats[j].min);
        stats[j].avg += local_stats[j].avg;
      }
    }
  }
  for (std::size_t j = 0; j < dataset_dim; j++) {
    stats[j].avg /= dataset_size;
  }

  // calc var
#pragma omp parallel
  {
    std::vector<detail::stat<T>> local_stats(dataset_dim);
    for (std::size_t i = omp_get_thread_num(); i < dataset_size;
         i += omp_get_num_threads()) {
      for (std::size_t j = 0; j < dataset_dim; j++) {
        const auto v = dataset_ptr[j + i * dataset_ld] - local_stats[j].avg;
        local_stats[j].var += v * v;
      }
    }
#pragma omp critical
    {
      for (std::size_t j = 0; j < dataset_dim; j++) {
        stats[j].var += local_stats[j].var;
      }
    }
  }
  for (std::size_t j = 0; j < dataset_dim; j++) {
    stats[j].var /= (dataset_size - 1);
  }

  // Print the result
  const std::uint32_t dimension_format_width =
      std::max(std::log10(dataset_dim) + 1, 3.);

  if (!graph_width) {
    std::printf("%*s | %7s, %7s, %7s, %7s\n", dimension_format_width, "dim",
                "avg", "var", "min", "max");
    for (std::size_t i = 0; i < dataset_size; i++) {
      std::printf("%*ld | %+4e, %+4e, %+4e, %+4e\n", stats[i].avg, stats[i].var,
                  static_cast<double>(stats[i].min),
                  static_cast<double>(stats[i].max));
    }
  } else {
    std::printf("%*s | %7s, %7s, %7s, %7s\n", dimension_format_width, "dim",
                "avg", "var", "min", "max");
    for (std::size_t i = 0; i < dataset_size; i++) {
      std::printf("%*ld | %+4e, %+4e, %+4e, %+4e\n", stats[i].avg, stats[i].var,
                  static_cast<double>(stats[i].min),
                  static_cast<double>(stats[i].max));
    }
  }
}
} // namespace mtk::anns_dataset
