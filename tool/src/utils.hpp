#pragma once
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#define UNUSED(a)                                                              \
  do {                                                                         \
    (void)(a);                                                                 \
  } while (0)

namespace mtk::anns_tool {
inline void print_info(const std::string str, const bool linebreak = true) {
  std::cout << "[ANNS tool INFO ]: " << str;
  if (linebreak) {
    std::cout << std::endl;
  }
}

inline void print_error(const std::string str, const bool linebreak = true) {
  std::cerr << "[ANNS tool ERROR]: " << str;
  if (linebreak) {
    std::cout << std::endl;
  }
}

inline void print_warning(const std::string str, const bool linebreak = true) {
  std::cerr << "[ANNS tool WARN ]: " << str;
  if (linebreak) {
    std::cout << std::endl;
  }
}

template <typename... Args>
std::string str_format(const std::string &fmt, Args... args) {
  const auto len = std::snprintf(nullptr, 0, fmt.c_str(), args...);
  std::vector<char> buf(len + 1);
  std::snprintf(buf.data(), len + 1, fmt.c_str(), args...);
  return std::string(buf.data(), buf.data() + len);
}

inline std::vector<std::string> split_str(const std::string params_str,
                                          const char sep) {
  std::stringstream ss{params_str};
  std::vector<std::string> res;
  std::string buffer;
  while (std::getline(ss, buffer, sep)) {
    res.push_back(buffer);
  }
  return res;
}

inline std::unordered_map<std::string, std::string>
parse_params(const std::string params_str) {
  const auto params = split_str(params_str, ',');
  std::unordered_map<std::string, std::string> res;

  for (const auto &p : params) {
    const auto pair = split_str(p, '=');
    if (pair.size() != 2) {
      print_warning("Value for key (" + pair[0] + ") not found");
    }
    res.insert(std::make_pair(pair[0], pair[1]));
  }
  return res;
}
} // namespace mtk::anns_tool
