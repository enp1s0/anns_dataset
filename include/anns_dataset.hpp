#pragma once
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>

namespace mtk {
namespace anns_dataset {
enum class format_t : std::uint32_t {
  FORMAT_UNKNOWN = 0,
  FORMAT_VECS = 0x1,
  FORMAT_BIGANN = 0x2,
  FORMAT_AUTO_DETECT = 0x4,
  HEADER_U32 = 0x100,
  HEADER_U64 = 0x200,

  FORMAT_MASK = 0xff,
  HEADER_MASK = 0xff00,
};

inline format_t operator|(const format_t a, const format_t b) {
  return static_cast<format_t>(static_cast<std::uint32_t>(a) |
                               static_cast<std::uint32_t>(b));
}
inline format_t operator&(const format_t a, const format_t b) {
  return static_cast<format_t>(static_cast<std::uint32_t>(a) &
                               static_cast<std::uint32_t>(b));
}

template <class HeaderT> inline format_t get_header_t();
template <> inline format_t get_header_t<std::uint32_t>() {
  return format_t::HEADER_U32;
}
template <> inline format_t get_header_t<std::uint64_t>() {
  return format_t::HEADER_U64;
}

inline std::string get_header_type_name(const format_t format) {
  switch (format & format_t::HEADER_MASK) {
  case format_t::HEADER_U64:
    return "u64";
  case format_t::HEADER_U32:
    return "u32";
  default:
    break;
  }
  return "Unknown";
}

inline std::string get_format_str(const format_t format) {
  std::string str;
  switch (format & format_t::FORMAT_MASK) {
  case format_t::FORMAT_VECS:
    str = "VECS";
    break;
  case format_t::FORMAT_BIGANN:
    str = "BIGANN";
    break;
  case format_t::FORMAT_UNKNOWN:
    str = "UNKNOWN";
    return str;
  case format_t::FORMAT_AUTO_DETECT:
    str = "AUTO_DETECT";
    return str;
  default:
    break;
  }
  return str + "(" + get_header_type_name(format) + ")";
}

namespace detail {
template <class data_T, class HEADER_T>
bool is_bigann(const HEADER_T header[2], const std::size_t file_size) {
  return static_cast<std::size_t>(header[0]) * header[1] * sizeof(data_T) +
             2 * sizeof(HEADER_T) ==
         file_size;
}
template <class data_T, class HEADER_T>
bool is_vecs(const HEADER_T header[2], const std::size_t file_size) {
  return (file_size % static_cast<std::size_t>(
                          sizeof(HEADER_T) + header[0] * sizeof(data_T))) == 0;
}
} // namespace detail

struct range_t {
  std::size_t offset;
  std::size_t size;
};

template <class T, class HEADER_T = void>
inline format_t detect_file_format(const std::string file_path,
                                   const bool print_log = false) {
  std::ifstream ifs(file_path);
  if (!ifs) {
    throw std::runtime_error("No such file: " + file_path);
  }

  if constexpr (std::is_same<HEADER_T, void>::value) {
    // HEADER_T auto detect
    if (print_log) {
      std::printf("[ANNS-DS %s]: Detecting HEADER_T...\n", __func__);
      std::fflush(stdout);
    }
    const auto v32 = detect_file_format<T, std::uint32_t>(file_path, print_log);
    if (v32 != mtk::anns_dataset::format_t::FORMAT_UNKNOWN)
      return v32;
    return detect_file_format<T, std::uint64_t>(file_path, print_log);
  } else {
    // Calculate file size
    ifs.seekg(0, ifs.end);
    const auto file_size = static_cast<std::size_t>(ifs.tellg());
    ifs.seekg(0, ifs.beg);

    HEADER_T header[2];
    ifs.read(reinterpret_cast<char *>(header), sizeof(header));
    ifs.close();

    const auto is_bigann = detail::is_bigann<T, HEADER_T>(header, file_size);
    const auto is_vecs = detail::is_vecs<T, HEADER_T>(header, file_size);

    mtk::anns_dataset::format_t format;
    if (is_bigann) {
      format = format_t::FORMAT_BIGANN | get_header_t<HEADER_T>();
    } else if (is_vecs) {
      format = format_t::FORMAT_VECS | get_header_t<HEADER_T>();
    } else {
      format = format_t::FORMAT_UNKNOWN;
    }

    if (print_log) {
      std::printf("[ANNS-DS %s]: Detected format = %s\n", __func__,
                  get_format_str(format).c_str());
      std::fflush(stdout);
    }
    return format;
  }
}

template <class T, class HEADER_T = void>
inline void load_size_info(const std::string file_path, std::size_t &num_data,
                           std::size_t &data_dim,
                           mtk::anns_dataset::format_t format =
                               mtk::anns_dataset::format_t::FORMAT_AUTO_DETECT,
                           const bool print_log = false) {
  num_data = data_dim = 0;
  if constexpr (std::is_same<HEADER_T, void>::value) {
    const auto detected_format =
        detect_file_format<T, void>(file_path, print_log);
    if (detected_format == format_t::FORMAT_UNKNOWN) {
      return;
    }

    const auto detected_header_t = detected_format & format_t::HEADER_MASK;
    const auto detected_format_t = detected_format & format_t::FORMAT_MASK;

    if (detected_header_t == format_t::HEADER_U32) {
      load_size_info<T, std::uint32_t>(file_path, num_data, data_dim,
                                       detected_format_t, print_log);
    } else {
      load_size_info<T, std::uint64_t>(file_path, num_data, data_dim,
                                       detected_format_t, print_log);
    }
  } else {
    num_data = 0;
    data_dim = 0;

    std::ifstream ifs(file_path);
    if (!ifs) {
      throw std::runtime_error("No such file: " + file_path);
    }

    if (print_log) {
      std::printf("[ANNS-DS %s]: Given format / mode = %s\n", __func__,
                  get_format_str(format).c_str());
      std::fflush(stdout);
    }

    // Calculate file size
    ifs.seekg(0, ifs.end);
    const auto file_size = static_cast<std::size_t>(ifs.tellg());
    ifs.seekg(0, ifs.beg);

    HEADER_T header[2];
    ifs.read(reinterpret_cast<char *>(header), sizeof(header));

    if (format == format_t::FORMAT_AUTO_DETECT) {
      if ((format = detect_file_format<T, HEADER_T>(file_path, print_log)) ==
          format_t::FORMAT_UNKNOWN) {
        throw std::runtime_error("Could not detect the file format: " +
                                 file_path);
      }
    }

    if ((format & format_t::FORMAT_VECS) != format_t::FORMAT_UNKNOWN) {
      data_dim = header[0];
      num_data = file_size / (sizeof(HEADER_T) + data_dim * sizeof(T));
    } else {
      data_dim = header[1];
      num_data = header[0];
    }
    ifs.close();
  }
}

template <class T, class HEADER_T = void>
inline std::pair<std::size_t, std::size_t>
load_size_info(const std::string file_path,
               const format_t format = format_t::FORMAT_AUTO_DETECT,
               const bool print_log = false) {
  std::size_t data_dim, num_data;

  load_size_info<T, HEADER_T>(file_path, num_data, data_dim, format, print_log);

  if (data_dim == 0 && num_data == 0) {
    throw std::runtime_error("No such file: " + file_path);
  }

  return std::make_pair(num_data, data_dim);
}

template <class MEM_T, class T = MEM_T, class HEADER_T = void>
int load(MEM_T *const ptr, const std::string file_path,
         const bool print_log = false,
         const format_t format = format_t::FORMAT_AUTO_DETECT,
         const range_t range = range_t{.offset = 0, .size = 0}) {
  if constexpr (std::is_same<HEADER_T, void>::value) {
    const auto detected_format =
        detect_file_format<T, void>(file_path, print_log);
    if (detected_format == format_t::FORMAT_UNKNOWN) {
      return 1;
    }

    const auto detected_header_t = detected_format & format_t::HEADER_MASK;
    const auto detected_format_t = detected_format & format_t::FORMAT_MASK;

    const auto f =
        format == format_t::FORMAT_AUTO_DETECT ? detected_format_t : format;
    if (detected_header_t == format_t::HEADER_U32) {
      load<MEM_T, T, std::uint32_t>(ptr, file_path, print_log, f, range);
    } else {
      load<MEM_T, T, std::uint64_t>(ptr, file_path, print_log, f, range);
    }
  } else {
    std::ifstream ifs(file_path);
    if (!ifs) {
      std::fprintf(stderr, "No such file : %s\n", file_path.c_str());
      return 1;
    }

    // Calculate file size
    ifs.seekg(0, ifs.end);
    const auto file_size = static_cast<std::size_t>(ifs.tellg());
    ifs.seekg(0, ifs.beg);

    if (print_log) {
      std::printf("[ANNS-DS %s]: Dataset path = %s\n", __func__,
                  file_path.c_str());
      std::printf("[ANNS-DS %s]: Dataset file size = %lu\n", __func__,
                  file_size);
      std::fflush(stdout);
    }

    HEADER_T header[2];
    ifs.read(reinterpret_cast<char *>(header), sizeof(header));

    format_t format_ = format;
    if (format == format_t::FORMAT_AUTO_DETECT) {
      if (detail::is_bigann<T, HEADER_T>(header, file_size)) {
        format_ = format_t::FORMAT_BIGANN;
      } else if (detail::is_vecs<T, HEADER_T>(header, file_size)) {
        format_ = format_t::FORMAT_VECS;
      } else {
        format_ = format_t::FORMAT_UNKNOWN;
        return 1;
      }
    }

    if (print_log) {
      std::printf("[ANNS-DS %s]: Format = ", __func__);
      if ((format_ & format_t::FORMAT_BIGANN) != format_t::FORMAT_UNKNOWN) {
        std::printf("FORMAT_BIGANN");
      } else if ((format_ & format_t::FORMAT_VECS) !=
                 format_t::FORMAT_UNKNOWN) {
        std::printf("FORMAT_VECS");
      }
      if (format == format_t::FORMAT_AUTO_DETECT) {
        std::printf(" (AUTO DETECTED)");
      }
      std::printf("\n");
    }

    constexpr auto loading_progress_interval = 1000;
    if (format_ == format_t::FORMAT_VECS) {
      const std::size_t data_dim = header[0];
      const std::size_t num_data =
          file_size / (sizeof(HEADER_T) + data_dim * sizeof(T));

      std::unique_ptr<T[]> buffer;
      if constexpr (!std::is_same<T, MEM_T>::value) {
        buffer = std::unique_ptr<T[]>(new T[data_dim]);
      }

      // Set load offset
      const auto num_load_vecs = range.size == 0 ? num_data : range.size;
      ifs.seekg(range.offset * (data_dim * sizeof(T) + sizeof(HEADER_T)),
                std::ios_base::beg);
      assert(num_load_vecs + range.offset <= num_data);

      if (print_log) {
        std::printf("[ANNS-DS %s]: Dataset dimension = %zu\n", __func__,
                    data_dim);
        std::printf("[ANNS-DS %s]: Num data = %zu\n", __func__, num_data);
        std::printf("[ANNS-DS %s]: Num load data = %zu, offset = %zu\n",
                    __func__, num_load_vecs, range.offset);
        std::fflush(stdout);
      }

      // Load
      for (HEADER_T i = 0; i < num_load_vecs; i++) {
        HEADER_T tmp;
        ifs.read(reinterpret_cast<char *>(&tmp), sizeof(HEADER_T));

        const auto offset = static_cast<std::uint64_t>(i) * data_dim;
        if constexpr (std::is_same<T, MEM_T>::value) {
          ifs.read(reinterpret_cast<char *>(ptr + offset),
                   sizeof(T) * data_dim);
        } else {
          ifs.read(reinterpret_cast<char *>(buffer.get()),
                   sizeof(T) * data_dim);
          for (std::uint32_t j = 0; j < data_dim; j++) {
            (ptr + offset)[j] = static_cast<MEM_T>(buffer.get()[j]);
          }
        }

        if (print_log) {
          if (num_load_vecs > loading_progress_interval &&
              i % (num_load_vecs / loading_progress_interval) == 0) {
            std::printf("[ANNS-DS %s]: Loading... (%4.2f %%)\r", __func__,
                        i * 100. / num_load_vecs);
            std::fflush(stdout);
          }
        }
      }
      if (print_log && num_load_vecs > loading_progress_interval) {
        std::printf("\n");
      }
    } else {
      const std::size_t data_dim = header[1];
      const std::size_t num_data = header[0];

      std::unique_ptr<T[]> buffer;
      if constexpr (!std::is_same<T, MEM_T>::value) {
        buffer = std::unique_ptr<T[]>(new T[data_dim]);
      }

      // Set load offset
      const auto num_load_vecs = range.size == 0 ? num_data : range.size;
      ifs.seekg(range.offset * data_dim * sizeof(T), std::ios_base::cur);
      assert(num_load_vecs + range.offset <= num_data);

      if (print_log) {
        std::printf("[ANNS-DS %s]: Dataset dimension = %zu\n", __func__,
                    data_dim);
        std::printf("[ANNS-DS %s]: Num data = %zu\n", __func__, num_data);
        std::printf("[ANNS-DS %s]: Num load data = %zu, offset = %zu\n",
                    __func__, num_load_vecs, range.offset);
        std::fflush(stdout);
      }

      // Load
      for (HEADER_T i = 0; i < num_load_vecs; i++) {
        const auto offset = static_cast<std::uint64_t>(i) * data_dim;
        if constexpr (std::is_same<T, MEM_T>::value) {
          ifs.read(reinterpret_cast<char *>(ptr + offset),
                   sizeof(T) * data_dim);
        } else {
          ifs.read(reinterpret_cast<char *>(buffer.get()),
                   sizeof(T) * data_dim);
          for (std::uint32_t j = 0; j < data_dim; j++) {
            (ptr + offset)[j] = static_cast<MEM_T>(buffer.get()[j]);
          }
        }
        if (print_log) {
          if (num_load_vecs > loading_progress_interval &&
              i % (num_load_vecs / loading_progress_interval) == 0) {
            std::printf("[ANNS-DS %s]: Loading... (%4.2f %%)\r", __func__,
                        i * 100. / num_load_vecs);
            std::fflush(stdout);
          }
        }
      }
      if (print_log && num_load_vecs > loading_progress_interval) {
        std::printf("\n");
      }
    }
    if (print_log) {
      std::printf("[ANNS-DS %s]: Completed\n", __func__);
      std::fflush(stdout);
    }
    ifs.close();
  }
  return 0;
}

template <class T> class store_stream {
  const std::size_t dataset_dim;
  format_t format;
  const bool print_log;

  std::ofstream ofs;
  std::size_t current_dataset_size_ = 0;

public:
  inline store_stream(const std::string dst_path, const std::size_t data_dim,
                      const format_t format, const bool print_log = false)
      : dataset_dim(data_dim), format(format), print_log(print_log) {
    ofs.open(dst_path, std::ios::binary);

    const auto format_t = format & format_t::FORMAT_MASK;
    const auto header_t = format & format_t::HEADER_MASK;
    if (format_t == mtk::anns_dataset::format_t::FORMAT_UNKNOWN) {
      throw std::runtime_error("[ANNS-DS store]: Unknown format (" +
                               get_format_str(format) + ")");
    }
    if (header_t == mtk::anns_dataset::format_t::FORMAT_UNKNOWN) {
      this->format = this->format | mtk::anns_dataset::format_t::HEADER_U32;
      if (print_log) {
        std::printf(
            "[ANNS-DS store]: Header type was not specified. Set to U32.\n");
      }
    }

    if (print_log) {
      std::printf("[ANNS-DS store]: Dataset path = %s\n", dst_path.c_str());
      std::printf("[ANNS-DS store]: Dataset dimension = %zu\n", data_dim);
      std::fflush(stdout);
    }
  }

private:
  template <class HEADER_T>
  inline void _append_core(const T *const dataset_ptr, const std::size_t ldd,
                           const std::size_t append_size) {
    current_dataset_size_ += append_size;
    const HEADER_T current_dataset_size = current_dataset_size_;

    if (print_log) {
      std::printf(
          "[ANNS-DS store]: Dataset append size = %zu, total size = %zu\n",
          append_size, current_dataset_size_);
      std::fflush(stdout);
    }

    constexpr auto loading_progress_interval = 1000;
    if ((format & format_t::FORMAT_VECS) != format_t::FORMAT_UNKNOWN) {
      for (std::size_t i = 0; i < append_size; i++) {
        const HEADER_T d = dataset_dim;
        ofs.write(reinterpret_cast<const char *>(&d), sizeof(HEADER_T));
        ofs.write(reinterpret_cast<const char *>(dataset_ptr + i * ldd),
                  sizeof(T) * dataset_dim);

        if (print_log) {
          if (append_size > loading_progress_interval &&
              i % (append_size / loading_progress_interval) == 0) {
            std::printf("[ANNS-DS store]: Storing... (%4.2f %%)\r",
                        i * 100. / append_size);
            std::fflush(stdout);
          }
        }
      }
    } else if ((format & format_t::FORMAT_BIGANN) != format_t::FORMAT_UNKNOWN) {
      const HEADER_T d = dataset_dim;
      const HEADER_T s = current_dataset_size;
      ofs.seekp(0, ofs.beg);
      ofs.write(reinterpret_cast<const char *>(&s), sizeof(HEADER_T));
      ofs.write(reinterpret_cast<const char *>(&d), sizeof(HEADER_T));

      ofs.seekp(0, ofs.end);
      for (std::size_t i = 0; i < append_size; i++) {
        ofs.write(reinterpret_cast<const char *>(dataset_ptr + i * ldd),
                  sizeof(T) * dataset_dim);

        if (print_log) {
          if (append_size > loading_progress_interval &&
              i % (append_size / loading_progress_interval) == 0) {
            std::printf("[ANNS-DS store]: Storing... (%4.2f %%)\r",
                        i * 100. / append_size);
            std::fflush(stdout);
          }
        }
      }
    }
    if (print_log) {
      if (append_size > loading_progress_interval) {
        std::printf("\n");
      }
      std::printf("[ANNS-DS store]: Completed\n");
      std::fflush(stdout);
    }
  }

public:
  inline void append(const T *const dataset_ptr, const std::size_t ldd,
                     const std::size_t append_size) {
    const auto header_t = format & format_t::HEADER_MASK;
    if (header_t == format_t::HEADER_U64) {
      this->template _append_core<std::uint64_t>(dataset_ptr, ldd, append_size);
    } else {
      this->template _append_core<std::uint32_t>(dataset_ptr, ldd, append_size);
    }
  }

  inline void close() { ofs.close(); }
};

template <class T>
inline int store(const std::string dst_path, const std::size_t data_size,
                 const std::size_t data_dim, const T *const data_ptr,
                 const format_t format, const bool print_log = false) {
  store_stream<T> ss(dst_path, data_dim, format, print_log);
  ss.append(data_ptr, data_dim, data_size);
  ss.close();

  return 0;
}
} // namespace anns_dataset
} // namespace mtk
