#pragma once
#include <iostream>
#include <fstream>

namespace mtk {
namespace anns_dataset {
enum format_t {
	TYPE_VECS,
	TYPE_BIGANN,
	TYPE_AUTO
};
template <class T>
inline format_t load_size_info(
		const std::string file_path,
		std::size_t& num_data,
		std::size_t& data_dim,
		format_t format
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

	// Calculate file size
	ifs.seekg(0, ifs.end);
	const auto file_size = static_cast<std::size_t>(ifs.tellg());
	ifs.seekg(0, ifs.beg);

	std::uint32_t header[2];
	ifs.read(reinterpret_cast<char*>(header), sizeof(header));

	const auto is_vecs = (static_cast<std::size_t>(header[0]) * header[1] * sizeof(T) + 2 * sizeof(std::uint32_t) != file_size);
	if (format == TYPE_AUTO) {
		if (is_vecs) {
			format = TYPE_VECS;
		} else {
			format = TYPE_BIGANN;
		}
	}

	if (format == TYPE_VECS) {
		data_dim = header[0];
		num_data = (file_size - sizeof(std::uint32_t)) / (sizeof(std::uint32_t) + data_dim * sizeof(T));
	} else {
		data_dim = header[1];
		num_data = header[0];
	}
	ifs.close();

	return format;
}

template <class T>
inline std::pair<std::size_t, std::size_t> load_size_info(
		const std::string file_path,
		const format_t format = TYPE_AUTO
		) {
	std::size_t data_dim, num_data;

	load_size_info<T>(file_path, num_data, data_dim, format);

	return std::make_pair(num_data, data_dim);
}

template <class T>
int load(
		T* const ptr,
		const std::string file_path,
		const bool print_log = false,
		const format_t format = TYPE_AUTO
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

	std::uint32_t header[2];
	ifs.read(reinterpret_cast<char*>(header), sizeof(header));

	const auto is_vecs = (static_cast<std::size_t>(header[0]) * header[1] * sizeof(T) + 2 * sizeof(std::uint32_t) != file_size);
	format_t format_ = format;
	if (format == TYPE_AUTO) {
		if (is_vecs) {
			format_ = TYPE_VECS;
		} else {
			format_ = TYPE_BIGANN;
		}
	}

	std::printf("[ANNS-DS %s]: Format = ", __func__);
	if (format_ == TYPE_BIGANN) {
		std::printf("TYPE_BIGANN");
	} else if (format_ == TYPE_VECS) {
		std::printf("TYPE_VECS");
	}
	if (format == TYPE_AUTO) {
		std::printf(" (AUTO DETECTED)");
	}
	std::printf("\n");

	if (format_ == TYPE_VECS) {
		const auto data_dim = header[0];
		const auto num_data = (file_size - sizeof(std::uint32_t)) / (sizeof(std::uint32_t) + data_dim * sizeof(T));
		if (print_log) {
			std::printf("[ANNS-DS %s]: Dataset dimension = %u\n", __func__, data_dim);
			std::printf("[ANNS-DS %s]: Num data = %u\n", __func__, num_data);
			std::fflush(stdout);
		}

		for (std::uint32_t i = 0; i < num_data; i++) {
			std::uint32_t tmp;
			ifs.read(reinterpret_cast<char*>(&tmp), sizeof(std::uint32_t));

			const auto offset = static_cast<std::uint64_t>(i) * data_dim;
			ifs.read(reinterpret_cast<char*>(ptr + offset), sizeof(T) * data_dim);

			if (print_log) {
				if (i % (num_data / 1000) == 0) {
					std::printf("[ANNS-DS %s]: Loading... (%4.2f %%)\r", __func__, i * 100. / num_data);
					std::fflush(stdout);
				}
			}
		}
		if (print_log) {
			std::printf("\n");
			std::fflush(stdout);
		}
	} else {
		const auto data_dim = header[1];
		const auto num_data = header[0];
		if (print_log) {
			std::printf("[ANNS-DS %s]: Dataset dimension = %u\n", __func__, data_dim);
			std::printf("[ANNS-DS %s]: Num data = %u\n", __func__, num_data);
			std::fflush(stdout);
		}

		for (std::uint32_t i = 0; i < num_data; i++) {
			const auto offset = static_cast<std::uint64_t>(i) * data_dim;
			ifs.read(reinterpret_cast<char*>(ptr + offset), sizeof(T) * data_dim);
			if (print_log) {
				if (i % (num_data / 1000) == 0) {
					std::printf("[ANNS-DS %s]: Loading... (%4.2f %%)\r", __func__, i * 100. / num_data);
					std::fflush(stdout);
				}
			}
		}
		if (print_log) {
			std::printf("\n");
			std::fflush(stdout);
		}
	}

	if (print_log) {
		std::fflush(stdout);
	}
	ifs.close();
	return 0;
}
} // namespace anns_dataset
} // namespace mtk
