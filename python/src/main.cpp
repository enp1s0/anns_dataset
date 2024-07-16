#include <anns_dataset.hpp>
#include <cstdint>
#include <exception>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>
#include <variant>
#include <vector>

enum class dtype_t {
  i32,
  u32,
  i8,
  u8,
  f32,
};

template <class T>
pybind11::array_t<T> load_core(const std::string filepath, const bool log) {
  std::size_t size = 0, dim = 0;
  mtk::anns_dataset::load_size_info<T>(filepath, size, dim);

  T *ptr = new T[size * dim];
  if (mtk::anns_dataset::load(ptr, filepath, log)) {
    throw std::runtime_error("Failed to load " + filepath);
  }

  pybind11::capsule destroy(ptr, [](void *f) {
    T *p = reinterpret_cast<T *>(f);
    delete[] p;
  });

  const auto buffer_info = pybind11::buffer_info(
      ptr,
      sizeof(T), // itemsize
      pybind11::format_descriptor<T>::format(),
      2,                                                   // ndim
      std::vector<std::size_t>{size, dim},                 // shape
      std::vector<std::size_t>{dim * sizeof(T), sizeof(T)} // strides
  );
  return pybind11::array_t<T>(buffer_info);
}

pybind11::object load(const std::string filepath, const dtype_t dtype,
                      const bool log) {
  if (dtype == dtype_t::i32) {
    return {load_core<std::int32_t>(filepath, log)};
  } else if (dtype == dtype_t::u32) {
    return {load_core<std::uint32_t>(filepath, log)};
  } else if (dtype == dtype_t::i8) {
    return {load_core<std::int8_t>(filepath, log)};
  } else if (dtype == dtype_t::u8) {
    return {load_core<std::uint8_t>(filepath, log)};
  } else if (dtype == dtype_t::f32) {
    return {load_core<float>(filepath, log)};
  }
  throw std::runtime_error("Unsupported dtype");

  return pybind11::array_t<float>{};
}

template <class T>
std::pair<std::size_t, std::size_t> get_shape_core(const std::string filepath) {
  std::size_t size = 0, dim = 0;
  mtk::anns_dataset::load_size_info<T>(filepath, size, dim);
  return std::pair<std::size_t, std::size_t>{size, dim};
}

std::pair<std::size_t, std::size_t> get_shape(const std::string filepath,
                                              const dtype_t dtype) {
  if (dtype == dtype_t::i32) {
    return get_shape_core<std::int32_t>(filepath);
  } else if (dtype == dtype_t::u32) {
    return get_shape_core<std::uint32_t>(filepath);
  } else if (dtype == dtype_t::i8) {
    return get_shape_core<std::int8_t>(filepath);
  } else if (dtype == dtype_t::u8) {
    return get_shape_core<std::uint8_t>(filepath);
  } else if (dtype == dtype_t::f32) {
    return get_shape_core<float>(filepath);
  }
  throw std::runtime_error("Unsupported dtype");

  return std::pair<std::size_t, std::size_t>{0, 0};
}

template <class T>
void store(pybind11::array_t<T> &buf, const std::string filepath,
           const mtk::anns_dataset::format_t format, const bool log) {
  pybind11::buffer_info buf_info = buf.request();
  std::size_t size, dim;
  if (buf_info.ndim == 2) {
    size = buf_info.shape[0];
    dim = buf_info.shape[1];
  } else {
    throw std::runtime_error("ndim must be 2 but " +
                             std::to_string(buf_info.ndim) + "is given.");
  }

  if (mtk::anns_dataset::store(filepath, size, dim,
                               static_cast<const T *>(buf_info.ptr), format,
                               log)) {
    throw std::runtime_error("Failed to save " + filepath);
  }
}

PYBIND11_MODULE(anns_dataset, m) {
  m.doc() = "anns_dataset_loader";

  m.def("load", &load, "", pybind11::arg("filepath"), pybind11::arg("dtype"),
        pybind11::arg("output_log") = false);
  m.def("store", &store<std::int32_t>, "", pybind11::arg("buffer"),
        pybind11::arg("filepath"), pybind11::arg("format"),
        pybind11::arg("output_log") = false);
  m.def("store", &store<std::uint32_t>, "", pybind11::arg("buffer"),
        pybind11::arg("filepath"), pybind11::arg("format"),
        pybind11::arg("output_log") = false);
  m.def("store", &store<std::int8_t>, "", pybind11::arg("buffer"),
        pybind11::arg("filepath"), pybind11::arg("format"),
        pybind11::arg("output_log") = false);
  m.def("store", &store<std::uint8_t>, "", pybind11::arg("buffer"),
        pybind11::arg("filepath"), pybind11::arg("format"),
        pybind11::arg("output_log") = false);
  m.def("store", &store<float>, "", pybind11::arg("buffer"),
        pybind11::arg("filepath"), pybind11::arg("format"),
        pybind11::arg("output_log") = false);
  m.def("get_shape", &get_shape, "", pybind11::arg("filepath"),
        pybind11::arg("dtype"));

  pybind11::enum_<mtk::anns_dataset::format_t>(m, "format_t")
      .value("FORMAT_VECS", mtk::anns_dataset::format_t::FORMAT_VECS)
      .value("FORMAT_BIGANN", mtk::anns_dataset::format_t::FORMAT_BIGANN)
      .export_values();

  pybind11::enum_<dtype_t>(m, "dtype_t")
      .value("u32", dtype_t::u32)
      .value("i32", dtype_t::i32)
      .value("u8", dtype_t::u8)
      .value("i8", dtype_t::i8)
      .value("f32", dtype_t::f32)
      .export_values();
}
