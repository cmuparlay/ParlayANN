#ifndef __DEBUG_HPP__
#define __DEBUG_HPP__

extern parlay::sequence<parlay::sequence<std::array<float,5>>> dist_in_search;
extern parlay::sequence<parlay::sequence<std::array<float,5>>> vc_in_search;
// extern parlay::sequence<uint32_t> round_in_search;
extern parlay::sequence<size_t> per_visited;
extern parlay::sequence<size_t> per_eval;
extern parlay::sequence<size_t> per_size_C;

#include <optional>

struct search_control{
	bool verbose_output;
	bool skip_search;
	float beta = 1;
	std::optional<float> radius;
	std::optional<uint32_t> log_per_stat;
	std::optional<uint32_t> log_dist;
	std::optional<uint32_t> log_size;
	std::optional<uint32_t> indicate_ep;
	std::optional<uint32_t> limit_eval;
	std::optional<uint32_t*> count_cmps;
};

#endif // _DEBUG_HPP_
