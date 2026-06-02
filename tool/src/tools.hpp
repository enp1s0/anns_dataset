#pragma once
#include <string>

namespace mtk::anns_tool {

struct arguments_t {
  enum class mode_t {
    none = 0,
    merge = 1,
    export_ = 2,
  } mode = mode_t::none;

  std::string input_path;
  std::string dtype;
  std::string output_path;

  std::string params;
};

int merge(mtk::anns_tool::arguments_t args);
} // namespace mtk::anns_tool
