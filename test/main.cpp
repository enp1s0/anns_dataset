#include <iostream>
#include <memory>
#include <anns_dataset_loader.hpp>

template <class data_t>
int test(
	const std::string dataset_path
	) {
	const auto [num_data, data_dim] = mtk::anns_dataset::load_size_info<data_t>(dataset_path);

	auto dataset_uptr = std::unique_ptr<data_t[]>(new data_t[num_data * data_dim]);

	if (mtk::anns_dataset::load(dataset_uptr.get(), dataset_path, true)) {
		std::printf("Failed\n");
		return 1;
	}
	std::printf("Succeeded\n");

	return 0;
}

int main(int argc, char** argv) {
	if (argc < 3) {
		std::printf("Usage: %s [/path/to/dataset] [dtype: uint8/float]\n", __FILE__);
		return 1;
	}
	const std::string dataset_path = argv[1];
	const std::string data_type = argv[2];

	if (data_type == "uint8") {
		return test<std::uint8_t>(dataset_path);
	} else if (data_type == "float") {
		return test<float>(dataset_path);
	}
	return 1;
}
