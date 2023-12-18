# ANNS Dataset loader

A header only file loading library for ANNS dataset such as SIFT1B, Yandex DEEP, and etc.

## Requirements
- C++ 17 or later

## Supported file formats
- `(num_data)(data_dim)(data_vector)*num_data`
  - e.g. BIGANN
- `(data_dim)(data_index, data_vector)*num_data`
  - e.g. ivecs, fvecs, etc

The input format is automatically detected.

## Sample
```cpp
// sample.cpp
// g++ -I/path/to/anns_dataset_loader/include sample.cpp ...
#include <anns_dataset_loader.hpp>

//...

const auto [num_data, data_dim] = mtk::anns_dataset::load_size_info<data_t>(dataset_path);

auto dataset_uptr = std::unique_ptr<data_t[]>(new data_t[num_data * data_dim]);

if (mtk::anns_dataset::load(dataset_uptr.get(), dataset_path, true)) {
  // load failed
}
```

## License
MIT
