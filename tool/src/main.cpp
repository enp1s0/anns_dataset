#include "tools.hpp"
#include "utils.hpp"
#include <argp.h>
#include <chrono>
#include <iostream>

static char doc[] = "ANN dataset tool";
static char args_doc[] = "[mode = merge]";

constexpr char ARG_OFFSET_SHORT = 0x80;
constexpr char ARG_SIZE_SHORT = 0x81;

static struct argp_option options[] = {
    {"input", 'i', "PATH", 0, "input datatse path"},
    {"dtype", 'd', "TYPE", 0, "data type = int8 | uint8 | float"},
    {"output", 'o', "PATH", 0, "output dataset path"},
    {"params", 'P', "STR", 0, "additional parameters"},
    {0}};

static error_t parse_opt(int key, char *arg, struct argp_state *state) {
  auto args =
      reinterpret_cast<struct mtk::anns_tool::arguments_t *>(state->input);
  switch (key) {
  case ARGP_KEY_ARG: {
    const auto mode_str = std::string(arg);
    if (mode_str == "merge") {
      args->mode = mtk::anns_tool::arguments_t::mode_t::merge;
    } else if (mode_str == "export") {
      args->mode = mtk::anns_tool::arguments_t::mode_t::export_;
    } else {
      argp_state_help(state, stderr, ARGP_HELP_STD_USAGE | ARGP_HELP_EXIT_ERR);
    }
  } break;
  case 'i':
    args->input_path = arg;
    break;
  case 'o':
    args->output_path = arg;
    break;
  case 'd':
    args->dtype = arg;
    break;
  case 'P':
    args->params = arg;
    break;
  case ARGP_KEY_END:
    if (args->mode == mtk::anns_tool::arguments_t::mode_t::none) {
      argp_state_help(state, stderr, ARGP_HELP_STD_USAGE | ARGP_HELP_EXIT_ERR);
    }
  default:
    return ARGP_ERR_UNKNOWN;
  }
  return 0;
}

int main(int argc, char **argv) {
  std::printf("+-------------------+\n"
              "| ANNS dataset tool |\n"
              "+-------------------+\n");
  mtk::anns_tool::arguments_t args;

  static struct argp argp = {options, parse_opt, args_doc, doc, 0, 0, 0};
  argp_parse(&argp, argc, argv, 0, 0, &args);

  const auto start_clock = std::chrono::system_clock::now();

  if (args.mode == mtk::anns_tool::arguments_t::mode_t::merge) {
    mtk::anns_tool::merge(args);
  }

  const auto end_clock = std::chrono::system_clock::now();
  const auto elapsed_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end_clock -
                                                            start_clock)
          .count() *
      1e-6;
  mtk::anns_tool::print_info(
      mtk::anns_tool::str_format("Duration: %e [s]", elapsed_time));
}
