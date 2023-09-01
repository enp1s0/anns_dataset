#pragma once
#include <iostream>
#include <fstream>
#include <memory>

namespace mtk {
namespace anns_dataset {
enum class format_t {
	FORMAT_VECS,
	FORMAT_BIGANN,
	FORMAT_AUTO_DETECT,
	FORMAT_UNKNOWN
};

inline std::string get_format_str(const format_t format) {
	switch (format) {
	case format_t::FORMAT_VECS: return "VECS";
	case format_t::FORMAT_BIGANN: return "BIGANN";
	case format_t::FORMAT_UNKNOWN: return "UNKNOWN";
	case format_t::FORMAT_AUTO_DETECT: return "AUTO_DETECT";
	default: break;
	}
	return "(unknown)";
}

namespace detail {
template <class data_T, class header_T>
bool is_bigann(const header_T header[2], const std::size_t file_size) {return static_cast<std::size_t>(header[0]) * header[1] * sizeof(data_T) + 2 * sizeof(header_T) == file_size;}
template <class data_T, class header_T>
bool is_vecs  (const header_T header[2], const std::size_t file_size) {return file_size % static_cast<std::size_t>(sizeof(header_T) + header[0] * sizeof(data_T)) == 0;}

} // unnamed namespace

template <class T, class header_T = std::uint32_t>
inline format_t detect_file_format(
		const std::string file_path,
		const bool print_log = false
		) {
	std::ifstream ifs(file_path);
	if (!ifs) {
		std::fprintf(
			stderr,
			"No such file : %s\n",
			file_path.c_str()
			);
	}

	// Calculate file size
	ifs.seekg(0, ifs.end);
	const auto file_size = static_cast<std::size_t>(ifs.tellg());
	ifs.seekg(0, ifs.beg);

	header_T header[2];
	ifs.read(reinterpret_cast<char*>(header), sizeof(header));
	ifs.close();

	const auto is_bigann = detail::is_bigann<T, header_T>(header, file_size);
	const auto is_vecs   = detail::is_vecs  <T, header_T>(header, file_size);

	mtk::anns_dataset::format_t format;
	if (is_bigann) {
		format = format_t::FORMAT_BIGANN;
	} else if (is_vecs) {
		format = format_t::FORMAT_VECS;
	} else {
		format = format_t::FORMAT_UNKNOWN;
	}

	if (print_log) {
		std::printf("[ANNS-DS %s]: Detected format = %s\n", __func__, get_format_str(format).c_str());
		std::fflush(stdout);
	}

	return format;
}

template <class T, class header_T = std::uint32_t>
inline void load_size_info(
		const std::string file_path,
		std::size_t& num_data,
		std::size_t& data_dim,
		format_t format = mtk::anns_dataset::format_t::FORMAT_AUTO_DETECT,
		const bool print_log = false
		) {
	num_data = 0;
	data_dim = 0;

	std::ifstream ifs(file_path);
	if (!ifs) {
		std::fprintf(
			stderr,
			"No such file : %s\n",
			file_path.c_str()
			);
	}

	if (print_log) {
		std::printf("[ANNS-DS %s]: Given format / mode = %s\n", __func__, get_format_str(format).c_str());
		std::fflush(stdout);
	}

	// Calculate file size
	ifs.seekg(0, ifs.end);
	const auto file_size = static_cast<std::size_t>(ifs.tellg());
	ifs.seekg(0, ifs.beg);

	header_T header[2];
	ifs.read(reinterpret_cast<char*>(header), sizeof(header));

	if (format == format_t::FORMAT_AUTO_DETECT) {
		if ((format = detect_file_format<T, header_T>(file_path, print_log)) == format_t::FORMAT_UNKNOWN) {
			return;
		}
	}

	if (format == format_t::FORMAT_VECS) {
		data_dim = header[0];
		num_data = file_size / (sizeof(header_T) + data_dim * sizeof(T));
	} else {
		data_dim = header[1];
		num_data = header[0];
	}
	ifs.close();
}

template <class T, class header_T = std::uint32_t>
inline std::pair<std::size_t, std::size_t> load_size_info(
		const std::string file_path,
		const format_t format = format_t::FORMAT_AUTO_DETECT,
		const bool print_log = false
		) {
	std::size_t data_dim, num_data;

	load_size_info<T, header_T>(file_path, num_data, data_dim, format, print_log);

	return std::make_pair(num_data, data_dim);
}

template <class MEM_T, class T = MEM_T, class HEADER_T = std::uint32_t>
int load(
		MEM_T* const ptr,
		const std::string file_path,
		const bool print_log = false,
		const format_t format = format_t::FORMAT_AUTO_DETECT
		) {

	std::ifstream ifs(file_path);
	if (!ifs) {
		std::fprintf(
			stderr,
			"No such file : %s\n",
			file_path.c_str()
			);
		return 1;
	}

	// Calculate file size
	ifs.seekg(0, ifs.end);
	const auto file_size = static_cast<std::size_t>(ifs.tellg());
	ifs.seekg(0, ifs.beg);

	if (print_log) {
		std::printf("[ANNS-DS %s]: Dataset path = %s\n", __func__, file_path.c_str());
		std::printf("[ANNS-DS %s]: Dataset file size = %lu\n", __func__, file_size);
		std::fflush(stdout);
	}

	HEADER_T header[2];
	ifs.read(reinterpret_cast<char*>(header), sizeof(header));

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

	std::printf("[ANNS-DS %s]: Format = ", __func__);
	if (format_ == format_t::FORMAT_BIGANN) {
		std::printf("FORMAT_BIGANN");
	} else if (format_ == format_t::FORMAT_VECS) {
		std::printf("FORMAT_VECS");
	}
	if (format == format_t::FORMAT_AUTO_DETECT) {
		std::printf(" (AUTO DETECTED)");
	}
	std::printf("\n");

	constexpr auto loading_progress_interval = 1000;
	if (format_ == format_t::FORMAT_VECS) {
		const std::size_t data_dim = header[0];
		const std::size_t num_data = file_size / (sizeof(HEADER_T) + data_dim * sizeof(T));
		if (print_log) {
			std::printf("[ANNS-DS %s]: Dataset dimension = %zu\n", __func__, data_dim);
			std::printf("[ANNS-DS %s]: Num data = %zu\n", __func__, num_data);
			std::fflush(stdout);
		}

		std::unique_ptr<T[]> buffer;
		if (!std::is_same<T, MEM_T>::value) {
			buffer = std::unique_ptr<T[]>(new T[data_dim]);
		}

		ifs.seekg(0, ifs.beg);
		for (HEADER_T i = 0; i < num_data; i++) {
			HEADER_T tmp;
			ifs.read(reinterpret_cast<char*>(&tmp), sizeof(HEADER_T));

			const auto offset = static_cast<std::uint64_t>(i) * data_dim;
			if (std::is_same<T, MEM_T>::value) {
				ifs.read(reinterpret_cast<char*>(ptr + offset), sizeof(T) * data_dim);
			} else {
				ifs.read(reinterpret_cast<char*>(buffer.get()), sizeof(T) * data_dim);
				for (std::uint32_t j = 0; j < data_dim; j++) {
					(ptr + offset)[j] = static_cast<MEM_T>(buffer.get()[j]);
				}
			}

			if (print_log) {
				if (num_data > loading_progress_interval && i % (num_data / loading_progress_interval) == 0) {
					std::printf("[ANNS-DS %s]: Loading... (%4.2f %%)\r", __func__, i * 100. / num_data);
					std::fflush(stdout);
				}
			}
		}
	} else {
		const std::size_t data_dim = header[1];
		const std::size_t num_data = header[0];
		if (print_log) {
			std::printf("[ANNS-DS %s]: Dataset dimension = %zu\n", __func__, data_dim);
			std::printf("[ANNS-DS %s]: Num data = %zu\n", __func__, num_data);
			std::fflush(stdout);
		}

		std::unique_ptr<T[]> buffer;
		if (!std::is_same<T, MEM_T>::value) {
			buffer = std::unique_ptr<T[]>(new T[data_dim]);
		}

		for (HEADER_T i = 0; i < num_data; i++) {
			const auto offset = static_cast<std::uint64_t>(i) * data_dim;
			if (std::is_same<T, MEM_T>::value) {
				ifs.read(reinterpret_cast<char*>(ptr + offset), sizeof(T) * data_dim);
			} else {
				ifs.read(reinterpret_cast<char*>(buffer.get()), sizeof(T) * data_dim);
				for (std::uint32_t j = 0; j < data_dim; j++) {
					(ptr + offset)[j] = static_cast<MEM_T>(buffer.get()[j]);
				}
			}
			if (print_log) {
				if (num_data > loading_progress_interval && i % (num_data / loading_progress_interval) == 0) {
					std::printf("[ANNS-DS %s]: Loading... (%4.2f %%)\r", __func__, i * 100. / num_data);
					std::fflush(stdout);
				}
			}
		}
	}
	if (print_log) {
		std::printf("\n");
		std::printf("[ANNS-DS %s]: Completed\n", __func__);
		std::fflush(stdout);
	}
	ifs.close();
	return 0;
}

template <class T, class header_T = std::uint32_t>
inline int store(
		const std::string dst_path,
		const std::size_t data_size,
		const std::size_t data_dim,
		const T* const data_ptr,
		const format_t format,
		const bool print_log = false
		) {
	std::ofstream ofs(dst_path);
	if (!ofs) {
		return 1;
	}
	if (print_log) {
		std::printf("[ANNS-DS %s]: Dataset path = %s\n", __func__, dst_path.c_str());
		std::printf("[ANNS-DS %s]: Dataset size = %zu\n", __func__, data_size);
		std::printf("[ANNS-DS %s]: Dataset dimension = %zu\n", __func__, data_dim);
		std::fflush(stdout);
	}

	constexpr auto loading_progress_interval = 1000;
	if (format == format_t::FORMAT_VECS) {
		for (std::size_t i = 0; i < data_size; i++) {
			const header_T d = data_dim;
			ofs.write(reinterpret_cast<const char*>(&d), sizeof(header_T));
			ofs.write(reinterpret_cast<const char*>(data_ptr + i * data_dim), sizeof(T) * data_dim);

			if (print_log) {
				if (data_size > loading_progress_interval && i % (data_size / loading_progress_interval) == 0) {
					std::printf("[ANNS-DS %s]: Loading... (%4.2f %%)\r", __func__, i * 100. / data_size);
					std::fflush(stdout);
				}
			}
		}
	} else if (format == format_t::FORMAT_BIGANN) {
		const header_T d = data_dim;
		const header_T s = data_size;
		ofs.write(reinterpret_cast<const char*>(&s), sizeof(header_T));
		ofs.write(reinterpret_cast<const char*>(&d), sizeof(header_T));

		for (std::size_t i = 0; i < data_size; i++) {
			ofs.write(reinterpret_cast<const char*>(data_ptr + i * data_dim), sizeof(T) * data_dim);

			if (print_log) {
				if (data_size > loading_progress_interval && i % (data_size / loading_progress_interval) == 0) {
					std::printf("[ANNS-DS %s]: Loading... (%4.2f %%)\r", __func__, i * 100. / data_size);
					std::fflush(stdout);
				}
			}
		}
	} else {
		std::printf("[ANNS-DS %s]: Unknown format (%s)\n", __func__, get_format_str(format).c_str());
		ofs.close();
		return 1;
	}
	if (print_log) {
		std::printf("\n");
		std::printf("[ANNS-DS %s]: Completed\n", __func__);
		std::fflush(stdout);
	}

	ofs.close();
	return 0;
}
} // namespace anns_dataset
} // namespace mtk
