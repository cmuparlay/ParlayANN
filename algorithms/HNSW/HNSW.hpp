#ifndef _HNSW_HPP
#define _HNSW_HPP

#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <cassert>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <memory>
#include <atomic>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <set>
#include <iterator>
#include <type_traits>
#include <limits>
#include <thread>
// #include "parallelize.h"
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/delayed_sequence.h>
#include <parlay/random.h>
#include "debug.hpp"
#include "../utils/beamSearch.h"
#define DEBUG_OUTPUT 0
#if DEBUG_OUTPUT
#define debug_output(...) fprintf(stderr, __VA_ARGS__)
#else
#define debug_output(...) do{[](...){}(__VA_ARGS__);}while(0)
#endif // DEBUG_OUTPUT

namespace ANN{

  using namespace parlayANN;
  
enum class type_metric{
	L2, ANGULAR, DOT
};

struct point{
	float x, y;
};

template<typename U, template<typename> class Allocator=std::allocator>
class HNSW
{
	using T = typename U::type_point;
	typedef uint32_t node_id;
public:
	/*
		Construct from the vectors [begin, end).
		std::iterator_trait<Iter>::value_type ought to be convertible to T
		dim: 				vector dimension
		m_l: 				control the # of levels (larger m_l leads to more layer)
		m: 					max degree
		ef_construction:	beam size during the construction
		alpha:				parameter of the heuristic (similar to the one in vamana)
		batch_base: 		growth rate of the batch size (discarded because of two passes)
	*/
	template<typename Iter>
	HNSW(Iter begin, Iter end, uint32_t dim, float m_l=1, uint32_t m=100, uint32_t ef_construction=50, float alpha=5, float batch_base=2);

	/*
		Construct from the saved model
		getter(i) returns the actual data (convertible to type T) of the vector with id i
	*/
	template<typename G>
	HNSW(const std::string &filename_model, G getter);

	parlay::sequence<std::pair<uint32_t,float>> search(
		const T &q, uint32_t k, uint32_t ef, const search_control &ctrl={}
	);
	// parlay::sequence<std::tuple<uint32_t,uint32_t,float>> search_ex(const T &q, uint32_t k, uint32_t ef, uint64_t verbose=0);
	// save the current model to a file
	void save(const std::string &filename_model) const;
public:
	typedef uint32_t type_index;

	struct node{
		// uint32_t id;
		uint32_t level;
		parlay::sequence<node_id> *neighbors;
		T data;
	};

	struct dist{
		float d;
		node_id u;

		constexpr bool operator<(const dist &rhs) const{
		return d<rhs.d;
		}

		constexpr bool operator>(const dist &rhs) const{
			return d>rhs.d;
		}
	};

	struct dist_ex : dist
	{
		uint32_t depth;
	};

	struct nearest{
		constexpr bool operator()(const dist &lhs, const dist &rhs) const{
			return lhs.d>rhs.d;
		}
	};

	struct farthest{
		constexpr bool operator()(const dist &lhs, const dist &rhs) const{
			return lhs.d<rhs.d;
		}
	};

/*
	struct cmp_id{
		constexpr bool operator()(const dist &lhs, const dist &rhs) const{
			return U::get_id(get_node(lhs.u).data)<U::get_id(get_node(rhs.u).data);
		}
	};
*/
	parlay::sequence<node_id> entrance; // To init
	// auto m, max_m0, m_L; // To init
	uint32_t dim;
	float m_l;
	uint32_t m;
	// uint32_t level_max = 30; // To init
	uint32_t ef_construction;
	float alpha;
	uint32_t n;
	Allocator<node> allocator;
	parlay::sequence<node> node_pool;
	mutable parlay::sequence<size_t> total_visited = parlay::sequence<size_t>(parlay::num_workers());
	mutable parlay::sequence<size_t> total_eval = parlay::sequence<size_t>(parlay::num_workers());
	mutable parlay::sequence<size_t> total_size_C = parlay::sequence<size_t>(parlay::num_workers());
	mutable parlay::sequence<size_t> total_range_candidate = parlay::sequence<size_t>(parlay::num_workers());

	static auto neighbourhood(node &u, uint32_t level)
		-> parlay::sequence<node_id>&
	{
		// const constexpr auto level_none = std::numeric_limits<uint32_t>::max();
		// return level==level_none? u.final_nbh: u.neighbors[level];
		// return level==0? u.final_nbh: u.neighbors[level];
		return u.neighbors[level];
	}

	static auto neighbourhood(const node &u, uint32_t level)
		-> const parlay::sequence<node_id>&
	{
		return neighbourhood(const_cast<node&>(u),level);
	}

	node& get_node(node_id id)
	{
		return node_pool[id];
	}

	const node& get_node(const node_id id) const
	{
		return node_pool[id];
	}

/*
	static void add_connection(parlay::sequence<node_id> &neighbors, node &u, uint32_t level)
	{
		for(auto pv : neighbors)
		{
			assert(&u!=pv);
			pv->neighbors[level].push_back(&u);
			u.neighbors[level].push_back(pv);
		}
	}
*/
	class dist_evaluator{
		using point_t = T;
		using dist_t = float;

		std::reference_wrapper<const point_t> p;
		uint32_t dim;
	public:
		dist_evaluator(const point_t &p, uint32_t dim) :
			p(p), dim(dim){
		}
		dist_t operator()(const point_t &pv) const{
			return U::distance(p, pv, dim);
		}
		dist_t operator()(const point_t &pu, const point_t &pv) const{
			return U::distance(pu, pv, dim);
		}
	};

	struct graph{
		template<class Nbh>
		struct edgeRange{
			edgeRange(Nbh &nbh) : nbh(nbh){
			}
			decltype(auto) operator[](node_id pu) const{
				return nbh.get()[pu];
			}
			auto size() const{
				return nbh.get().size();
			}
			void prefetch() const{
				int l = (size() * sizeof(node_id))/64;
				for (int i=0; i < l; i++)
					__builtin_prefetch((char*) nbh.get().data() + i*64);
			}

			std::reference_wrapper<Nbh> nbh;
		};

		using nid_t = node_id;

		graph(const HNSW<U,Allocator> &hnsw, uint32_t l) :
			hnsw(hnsw), l(l){
		}

		decltype(auto) num_nodes() const{
			return hnsw.get().n;
		}
		decltype(auto) get_node(node_id pu) const{
			return hnsw.get().get_node(pu);
		}
		decltype(auto) get_edges(node_id pu){
			return hnsw.get().neighbourhood(hnsw.get().get_node(pu),l);
		}
		decltype(auto) get_edges(node_id pu) const{
			return hnsw.get().neighbourhood(hnsw.get().get_node(pu),l);
		}

		uint32_t max_degree() const{
			return hnsw.get().get_threshold_m(l);
		}

		auto operator[](node_id pu){
			return edgeRange(get_edges(pu));
		}
		auto operator[](node_id pu) const{
			return edgeRange(get_edges(pu));
		}

		std::reference_wrapper<const HNSW<U,Allocator>> hnsw;
		uint32_t l;
	};

	// node* insert(const T &q, uint32_t id);
	template<typename Iter>
	void insert(Iter begin, Iter end, bool from_blank);

	template<typename Queue>
	void select_neighbors_simple_impl(const T &u, Queue &C, uint32_t M)
	{
		/*
		list res;
		for(uint32_t i=0; i<M; ++i)
		{
			res.insert(C.pop_front());
		}
		return res;
		*/
		(void)u;
		parlay::sequence<typename Queue::value_type> tie;
		float dist_tie = 1e20;
		while(C.size()>M)
		{
			const auto &t = C.top();
			if(t.d+1e-6<dist_tie) // t.d<dist_tie
			{
				dist_tie = t.d;
				tie.clear();
			}
			if(fabs(dist_tie-t.d)<1e-6) // t.d==dist_tie
				tie.push_back(t);
			C.pop();
		}
		if(fabs(dist_tie-C.top().d)<1e-6) // C.top().d==dist_tie
			while(!tie.empty())
			{
			//	C.push({dist_tie,tie.back()});
				C.push(tie.back());
				tie.pop_back();
			}
	}

	template<typename Queue>
	auto select_neighbors_simple(const T &u, const Queue &C, uint32_t M)
	{
		// The parameter C is intended to be copy constructed
		/*
		select_neighbors_simple_impl(u, C, M);
		return C;
		*/
		// auto R = parlay::sort(C, farthest());
		auto R = C;
		
		if(R.size()>M)
		{
			std::nth_element(R.begin(), R.begin()+M, R.end(), farthest());
			R.resize(M);
		}
		
		std::sort(R.begin(), R.end(), farthest());
		// if(R.size()>M) R.resize(M);
		/*
		uint32_t size_R = std::min(C.size(),M);
		parlay::sequence<node*> R;
		R.reserve(size_R);
		for(const auto &e : C)
			R.push_back(e.u);
		*/

		return R;
	}

	// To optimize
	auto select_neighbors_heuristic(const T &u, 
		/*const std::priority_queue<dist,parlay::sequence<dist>,farthest> &C*/
		const parlay::sequence<dist> &C, uint32_t M,
		uint32_t level, bool extendCandidate, bool keepPrunedConnections)
	{
		(void)extendCandidate;

		// std::priority_queue<dist,parlay::sequence<dist>,farthest> C_cp=C, W_d;
		parlay::sequence<dist> W_d;
		std::set<node_id> W_tmp;
		// while(!C_cp.empty())
		for(auto &e : C) // TODO: add const?
		{
			// auto &e = C_cp.top();
			W_tmp.insert(e.u);
			if(extendCandidate)
			{
				for(node_id e_adj : neighbourhood(get_node(e.u),level))
				{
					// if(e_adj==nullptr) continue; // TODO: check
					if(W_tmp.find(e_adj)==W_tmp.end())
						W_tmp.insert(e_adj);
				}
			}
			// C_cp.pop();
		}

		// std::priority_queue<dist,parlay::sequence<dist>,nearest> W;
		parlay::sequence<dist> W;
		W.reserve(W_tmp.size());
		for(node_id p : W_tmp)
			W.push_back({U::distance(get_node(p).data,u,dim), p});
		std::sort(W.begin(), W.end(), farthest());
		/*
		for(auto &e : W_tmp)
			W.push(e);
		*/
		W_tmp.clear();

		parlay::sequence<node_id> R;
		std::set<node_id> nbh;
		// while(W.size()>0 && R.size()<M)
		for(const auto &e : W)
		{
			if(R.size()>=M) break;
			// const auto e = W.top();
			// W.pop();
			const auto d_q = e.d;

			bool is_good = true;
			for(const auto &r : R)
			{
				const auto d_r = U::distance(get_node(e.u).data, get_node(r).data, dim);
				//if(d_r*(level+1)>d_q*alpha*(entrance->level+1))
				if(d_r<d_q*alpha)
				{
					is_good = false;
					break;
				}
				/*
				for(auto *pv : neighbourhood(*e.u,level))
					if(pv==e.u)
					{
						is_good = false;
						break;
					}
				*/
				/*
				if(nbh.find(e.u)!=nbh.end())
					is_good = false;
				*/
			}

			if(is_good)
			{
				R.push_back(e.u);
				/*				
				for(auto *pv : neighbourhood(*e.u,level))
					nbh.insert(pv);
				*/
			}
			else
				W_d.push_back(e);
		}

		// std::sort(W_d.begin(), W_d.end(), nearest());
		auto it = W_d.begin();
		// std::priority_queue<dist,parlay::sequence<dist>,farthest> res;
		auto &res = R;
		/*
		for(const auto &r : R)
		{
			res.push({U::distance(u,get_node(r).data,dim), r});
		}
		*/
		if(keepPrunedConnections)
		{
			// while(W_d.size()>0 && res.size()<M)
				// res.push(W_d.top()), W_d.pop();
			while(it!=W_d.end() && res.size()<M)
				// res.push(*(it++));
				res.push_back((it++)->u);
		}
		return res;
	}

	template<class Seq_, class D, class G, class Seq=std::remove_cv_t<std::remove_reference_t<Seq_>>>
	Seq prune_heuristic(
		Seq_ &&cand, uint32_t size, D f_dist, G g) const
	{
		using nid_t = node_id;
		using conn = dist;

		Seq workset = std::forward<Seq_>(cand);
		/*
		if(ctrl.extend_nbh)
		{
			const auto &g = ctrl.graph;
			std::unordered_set<nid_t> cand_ext;
			for(const conn &c : workset)
			{
				cand_ext.insert(c.u);
				for(nid_t pv : g.get_edges(c.u))
					cand_ext.insert(pv);
			}

			workset.reserve(workset.size()+cand_ext.size());
			for(nid_t pc : cand_ext)
				workset.push_back({f_dist(g.get_node(pc).get_coord()), pc});
			cand_ext.clear();
		}
		*/
		parlay::sort_inplace(workset);

		Seq res, pruned;
		std::unordered_set<nid_t> nbh;
		for(conn &c : workset)
		{
			const auto d_cu = c.d*alpha;

			bool is_pruned = false;
			for(const conn &r : res)
			{
				const auto d_cr = f_dist(
					g.get_node(c.u).data,
					g.get_node(r.u).data
				);
				if(d_cr<d_cu)
				{
					is_pruned = true;
					break;
				}
			}

			if(!is_pruned)
			{
				res.push_back(std::move(c));
				if(res.size()==size) break;
			}
			else pruned.push_back(std::move(c));
		}
		return res;
	}

	auto select_neighbors(const T &u, 
		/*const std::priority_queue<dist,parlay::sequence<dist>,farthest> &C,*/
		const parlay::sequence<dist> &C, uint32_t M,
		uint32_t level, bool extendCandidate=false, bool keepPrunedConnections=false)
	{
		/*
		(void)level, (void)extendCandidate, (void)keepPrunedConnections;
		return select_neighbors_simple(u,C,M);
		*/
		// return select_neighbors_heuristic(u, C, M, level, extendCandidate, keepPrunedConnections);

		dist_evaluator f_dist(u, dim);
		graph g(*this, level);
		auto res = prune_heuristic(C, M, f_dist, g);
		return parlay::tabulate(res.size(), [&](size_t i){return res[i].u;});
	}

	uint32_t get_level_random()
	{
		// static thread_local int32_t anchor;
		// uint32_t esp;
		// asm volatile("movl %0, %%esp":"=a"(esp));
		// static thread_local std::hash<std::thread::id> h;
		// static thread_local std::mt19937 gen{h(std::this_thread::get_id())};
		static thread_local std::mt19937 gen{parlay::worker_id()};
		static thread_local std::uniform_real_distribution<> dis(std::numeric_limits<float>::min(), 1.0);
		const uint32_t res = uint32_t(-log(dis(gen))*m_l);
		return res;
	}

	// auto search_layer(const node &u, const parlay::sequence<node_id> &eps, uint32_t ef, uint32_t l_c, uint64_t verbose=0) const; // To static
	auto search_layer(const node &u, const parlay::sequence<node_id> &eps, uint32_t ef, uint32_t l_c, search_control ctrl={}) const; // To static
	auto search_layer_bak(const node &u, const parlay::sequence<node_id> &eps, uint32_t ef, uint32_t l_c, search_control ctrl={}) const; // To static
	auto search_layer_new_ex(const node &u, const parlay::sequence<node_id> &eps, uint32_t ef, uint32_t l_c, search_control ctrl={}) const; // To static
	auto beam_search_ex(const node &u, const parlay::sequence<node_id> &eps, uint32_t beamSize, uint32_t l_c, search_control ctrl={}) const;
	parlay::sequence<node_id> search_layer_to(
		const node &u, uint32_t ef, uint32_t l_stop, const search_control &ctrl={}
	);

	auto get_threshold_m(uint32_t level) const{
		return level==0? m*2: m;
		// (void)level;
		// return m;
	}

public:
	auto get_deg(uint32_t level=0)
	{
		parlay::sequence<uint32_t> res;
		res.reserve(node_pool.size());
		for(const node &e : node_pool)
		{
			if(e.level>=level)
				res.push_back(e.neighbors[level].size());
		}
		return res;
	}

	auto get_indeg(uint32_t level) const
	{
		static uint32_t *indeg[16] = {nullptr};
		auto *&res = indeg[level];
		if(!res)
		{
			res = new uint32_t[n];
			for(uint32_t i=0; i<n; ++i)
				res[i] = 0;
			for(const node_id pu : node_pool)
			{
				if(get_node(pu).level<level) continue;
				for(const node_id pv : get_node(pu).neighbors[level])
					res[U::get_id(get_node(pv).data)]++;
			}
		}
		return res;
	}

	uint32_t get_height() const
	{
		return get_node(entrance[0]).level;
	}

	size_t cnt_degree(uint32_t l) const
	{
		auto cnt_each = parlay::delayed_seq<size_t>(n, [&](size_t i){
			node_id pu = i;
			return get_node(pu).level<l? 0:
				neighbourhood(get_node(pu),l).size();
		});
		return parlay::reduce(cnt_each, parlay::addm<size_t>());
	}

	size_t cnt_vertex(uint32_t l) const
	{
		auto cnt_each = parlay::delayed_seq<size_t>(n, [&](size_t i){
			node_id pu = i;
			return get_node(pu).level<l? 0: 1;
		});
		return parlay::reduce(cnt_each, parlay::addm<size_t>());
	}

	size_t get_degree_max(uint32_t l) const
	{
		auto cnt_each = parlay::delayed_seq<size_t>(n, [&](size_t i){
			node_id pu = i;
			return get_node(pu).level<l? 0:
				neighbourhood(get_node(pu),l).size();
		});
		return parlay::reduce(cnt_each, parlay::maxm<size_t>());
	}
/*
	void debug_output_graph(uint32_t l)
	{
		// return;
		debug_output("Printing the graph at level %u\n", l);
		auto node_exist = parlay::pack(
			node_pool,
			parlay::delayed_seq<bool>(node_pool.size(),[&](size_t i){
				return node_pool[i]->level>=l;
			})
		);
		const auto num_vertices = node_exist.size();
		const auto num_edges = parlay::reduce(
			parlay::delayed_seq<uint64_t>(node_exist.size(),[&](size_t i){
				return node_exist[i]->neighbors[l].size();
			}),
			parlay::addm<uint64_t>{}
		);
		debug_output("# vertices: %lu, # edges: %llu\n", num_vertices, num_edges);

		for(node_id pu : node_exist)
		{
			debug_output("node_id: %u\n", U::get_id(get_node(pu).data));
			// if(!res[i]) continue;
			debug_output("\tneighbors:");
			for(node_id pv : neighbourhood(get_node(pu),l))
				debug_output(" %u", U::get_id(get_node(pv).data));
			debug_output("\n");
		}
	}
*/
};

template<typename U, template<typename> class Allocator>
template<typename G>
HNSW<U,Allocator>::HNSW(const std::string &filename_model, G getter)
{
	std::ifstream model(filename_model, std::ios::binary);
	if(!model.is_open())
		throw std::runtime_error("Failed to open the model");

	const auto size_buffer = 1024*1024*1024; // 1G
	auto buffer = std::make_unique<char[]>(size_buffer);
	model.rdbuf()->pubsetbuf(buffer.get(), size_buffer);

	auto read = [&](auto &data, auto ...args){
		auto read_impl = [&](auto &f, auto &data, auto ...args){
			using T = std::remove_reference_t<decltype(data)>;
			if constexpr(std::is_pointer_v<std::decay_t<T>>)
			{
				auto read_array = [&](auto &data, size_t size, auto ...args){
					for(size_t i=0; i<size; ++i)
						f(f, data[i], args...);
				};
				// use the array extent as the size
				if constexpr(sizeof...(args)==0 && std::is_array_v<T>)
				{
					read_array(data, std::extent_v<T>);
				}
				else
				{
					static_assert(sizeof...(args), "size was not provided");
					read_array(data, args...);
				}
			}
			else
			{
				static_assert(std::is_standard_layout_v<T>);
				model.read((char*)&data, sizeof(data));
			}
		};
		read_impl(read_impl, data, args...);
	};

	char model_type[5] = {'\000'};
	read(model_type, 4);
	if(strcmp(model_type,"HNSW"))
		throw std::runtime_error("Wrong type of model");
	uint32_t version;
	read(version);
	if(version!=3)
		throw std::runtime_error("Unsupported version");

	size_t code_U, size_node;
	read(code_U);
	read(size_node);
	fprintf(stderr, "U type loading %s\n", typeid(U).name());
	// if((typeid(U).hash_code()^sizeof(U))!=code_U)
		// throw std::runtime_error("Inconsistent type `U`");
	// if(sizeof(node)!=size_node)
		// throw std::runtime_error("Inconsistent type `node`");

	// read parameter configuration
	read(dim);
	read(m_l);
	read(m);
	read(ef_construction);
	read(alpha);
	read(n);
	puts("Configuration loaded");
	printf("dim = %u\n", dim);
	printf("m_l = %f\n", m_l);
	printf("m = %u\n", m);
	printf("efc = %u\n", ef_construction);
	printf("alpha = %f\n", alpha);
	printf("n = %u\n", n);
	// read indices
	// std::unordered_map<uint32_t,node*> addr;
	node_pool.resize(n);
	for(uint32_t i=0; i<n; ++i)
	{
		// auto *u = new node;
		node &u = get_node(i);
		read(u.level);
		uint32_t id_u; // TODO: use generic type
		read(id_u);
		u.data = getter(id_u);
		// addr[id_u] = u;
	}
	for(node &u : node_pool)
	{
		u.neighbors = new parlay::sequence<node_id>[u.level+1];
		for(uint32_t l=0; l<=u.level; ++l)
		{
			size_t size;
			read(size);
			auto &nbh_u = u.neighbors[l];
			nbh_u.reserve(size);
			for(size_t i=0; i<size; ++i)
			{
				uint32_t id_v;
				read(id_v);
				nbh_u.push_back(id_v);
			}
		}
	}
	// read entrances
	size_t size;
	read(size);
	entrance.reserve(size);
	for(size_t i=0; i<size; ++i)
	{
		uint32_t id_u;
		read(id_u);
		entrance.push_back(id_u);
	}
}

template<typename U, template<typename> class Allocator>
template<typename Iter>
HNSW<U,Allocator>::HNSW(Iter begin, Iter end, uint32_t dim_, float m_l_, uint32_t m_, uint32_t ef_construction_, float alpha_, float batch_base)
	: dim(dim_), m_l(m_l_), m(m_), ef_construction(ef_construction_), alpha(alpha_), n(std::distance(begin,end))
{
	static_assert(std::is_same_v<typename std::iterator_traits<Iter>::value_type, T>);
	static_assert(std::is_base_of_v<
		std::random_access_iterator_tag, typename std::iterator_traits<Iter>::iterator_category>);

	if(n==0) return;

	std::random_device rd;
	auto perm = parlay::random_permutation<uint32_t>(n, rd());
	auto rand_seq = parlay::delayed_seq<T>(n, [&](uint32_t i){
		//return *(begin+perm[i]);
		return *(begin+i);
	});

	const auto level_ep = get_level_random();
	// node *entrance_init = allocator.allocate(1);
	// node_pool.push_back(entrance_init);
	node_pool.resize(1);
	node_id entrance_init = 0;
	new(&get_node(entrance_init)) node{
		level_ep, 
		new parlay::sequence<node_id>[level_ep+1], 
		*rand_seq.begin()
		/*anything else*/
	};
	entrance.push_back(entrance_init);

	uint32_t batch_begin=0, batch_end=1, size_limit=n*0.02;
	float progress = 0.0;
	while(batch_end<n)
	{
		batch_begin = batch_end;
		batch_end = std::min({n, (uint32_t)std::ceil(batch_begin*batch_base)+1, batch_begin+size_limit});
		/*
		if(batch_end>batch_begin+100)
			batch_end = batch_begin+100;
		*/
		// batch_end = batch_begin+1;

		insert(rand_seq.begin()+batch_begin, rand_seq.begin()+batch_end, true);
		// insert(rand_seq.begin()+batch_begin, rand_seq.begin()+batch_end, false);

		if(batch_end>n*(progress+0.05))
		{
			progress = float(batch_end)/n;
			fprintf(stderr, "Built: %3.2f%%\n", progress*100);
			// fprintf(stderr, "# visited: %lu\n", parlay::reduce(total_visited,parlay::addm<size_t>{}));
			// fprintf(stderr, "# eval: %lu\n", parlay::reduce(total_eval,parlay::addm<size_t>{}));
			// fprintf(stderr, "size of C: %lu\n", parlay::reduce(total_size_C,parlay::addm<size_t>{}));
		}
	}

	// fprintf(stderr, "# visited: %lu\n", parlay::reduce(total_visited,parlay::addm<size_t>{}));
	// fprintf(stderr, "# eval: %lu\n", parlay::reduce(total_eval,parlay::addm<size_t>{}));
	// fprintf(stderr, "size of C: %lu\n", parlay::reduce(total_size_C,parlay::addm<size_t>{}));
	fprintf(stderr, "Index built\n");

	#if 0
		for(const auto *pu : node_pool)
		{
			fprintf(stderr, "[%u] (%.2f,%.2f)\n", U::get_id(get_node(pu).data), get_node(pu).data[0], get_node(pu).data[1]);
			for(int32_t l=pu->level; l>=0; --l)
			{
				fprintf(stderr, "\tlv. %d:", l);
				for(const auto *k : pu->neighbors[l])
					fprintf(stderr, " %u", U::get_id(get_node(k).data));
				fputs("\n", stderr);
			}
		}
	#endif
/*
	for(uint32_t l=0; l<entrance[0]->level; ++l)
		debug_output_graph(l);
*/
}

template<typename U, template<typename> class Allocator>
template<typename Iter>
void HNSW<U,Allocator>::insert(Iter begin, Iter end, bool from_blank)
{
	const auto level_ep = get_node(entrance[0]).level;
	const auto size_batch = std::distance(begin,end);
	auto node_new = std::make_unique<node_id[]>(size_batch);
	auto nbh_new = std::make_unique<parlay::sequence<node_id>[]>(size_batch);
	auto eps = std::make_unique<parlay::sequence<node_id>[]>(size_batch);
	//const float factor_m = from_blank? 0.5: 1;
	const auto factor_m = 1;

	debug_output("Insert %lu elements; from blank? [%c]\n", size_batch, "NY"[from_blank]);

	// auto *pool = allocator.allocate(size_batch);
	// first, query the nearest point as the starting point for each node to insert
	if(from_blank)
	{
		auto offset = node_pool.size();
		node_pool.resize(offset+size_batch);
	parlay::parallel_for(0, size_batch, [&](uint32_t i){
		const T &q = *(begin+i);
		const auto level_u = get_level_random();
		// auto *const pu = &pool[i];		// TODO: add pointer manager
		node_id pu = offset+i;

		new(&get_node(pu)) node{
			level_u,
			new parlay::sequence<node_id>[level_u+1],
			q
		};
		node_new[i] = pu;
	});
	}
	else
	{
	parlay::parallel_for(0, size_batch, [&](uint32_t i){
		node_new[i] = node_pool.size()-size_batch+i;
	});
	}

	debug_output("Nodes are settled\n");
	// TODO: merge ops
	parlay::parallel_for(0, size_batch, [&](uint32_t i){
		auto &u = get_node(node_new[i]);
		const auto level_u = u.level;
		auto &eps_u = eps[i]; 
		// eps_u.push_back(entrance);
		eps_u = entrance;
		for(uint32_t l=level_ep; l>level_u; --l)
		{
			const auto res = search_layer(u, eps_u, 1, l); // TODO: optimize
			eps_u.clear();
			eps_u.push_back(res[0].u);
		}
	});

	debug_output("Finish searching entrances\n");
	// then we process them layer by layer (from high to low)
	for(int32_t l_c=level_ep; l_c>=0; --l_c) // TODO: fix the type
	{
		parlay::sequence<parlay::sequence<std::pair<node_id,node_id>>> edge_add(size_batch);

		debug_output("Finding neighbors on lev. %d\n", l_c);
		parlay::parallel_for(0, size_batch, [&](uint32_t i){
			node_id pu = node_new[i];
			auto &u = get_node(pu);
			if((uint32_t)l_c>u.level) return;

			auto &eps_u = eps[i]; // TODO: check
			auto res = search_layer(u, eps_u, ef_construction, l_c);
			auto neighbors_vec = select_neighbors(u.data, res, get_threshold_m(l_c)/**factor_m*/, l_c);
			// move the content from `neighbors_vec` to `u.neighbors[l_c]`
			// auto &nbh_u = nbh_new[i];
			auto &edge_u = edge_add[i];
			// nbh_u.clear();
			edge_u.clear();
			// nbh_u.reserve(neighbors_vec.size());
			edge_u.reserve(neighbors_vec.size());
			/*
			for(uint32_t j=0; neighbors_vec.size()>0; ++j)
			{
				auto *pv = neighbors_vec.top().u;
				neighbors_vec.pop();
				// nbh_u[j] = pv;
				// edge_u[j] = std::make_pair(pv, &u);
				nbh_u.push_back(pv);
				edge_u.emplace_back(pv, &u);
			}
			*/
			for(node_id pv : neighbors_vec)
				edge_u.emplace_back(pv, pu);
			nbh_new[i] = std::move(neighbors_vec);

			eps_u.clear();
			/*
			while(res.size()>0)
			{
				eps_u.push_back(res.top().u); // TODO: optimize
				res.pop();
			}
			*/
			eps_u.reserve(res.size());
			for(const auto e : res)
				eps_u.push_back(e.u);
		});

		debug_output("Adding forward edges\n");
		parlay::parallel_for(0, size_batch, [&](uint32_t i){
			auto &u = get_node(node_new[i]);
			if((uint32_t)l_c<=u.level)
				neighbourhood(u,l_c) = std::move(nbh_new[i]);
		});

		debug_output("Adding reverse edges\n");
		// now we add edges in the other direction
		auto edge_add_flatten = parlay::flatten(edge_add);
		auto edge_add_grouped = parlay::group_by_key(edge_add_flatten);

		parlay::parallel_for(0, edge_add_grouped.size(), [&](size_t j){
			node_id pv = edge_add_grouped[j].first;
			auto &nbh_v = neighbourhood(get_node(pv),l_c);
			auto &nbh_v_add = edge_add_grouped[j].second;

			// std::unordered_set<node_id> hash_table(nbh_v.begin(),nbh_v.end());
			/*
			for(auto it=nbh_v_add.begin(); it!=nbh_v_add.end();)
			{
				bool is_extant = *it==pv||std::find_if(nbh_v.begin(), nbh_v.end(), [&](const node_id pu_extant){
					return *it==pu_extant;
				})!=nbh_v.end();
				
				// bool is_extant = hash_table.find(*it)!=hash_table.end();
				it = is_extant? nbh_v_add.erase(it): std::next(it);
			}
			*/

			const uint32_t size_nbh_total = nbh_v.size()+nbh_v_add.size();

			const auto m_s = get_threshold_m(l_c)*factor_m;
			if(size_nbh_total>m_s)
			{
				auto candidates = parlay::sequence<dist>(size_nbh_total);
				for(size_t k=0; k<nbh_v.size(); ++k)
					candidates[k] = dist{U::distance(get_node(nbh_v[k]).data,get_node(pv).data,dim), nbh_v[k]};
				for(size_t k=0; k<nbh_v_add.size(); ++k)
					candidates[k+nbh_v.size()] = dist{U::distance(get_node(nbh_v_add[k]).data,get_node(pv).data,dim), nbh_v_add[k]};

				std::sort(candidates.begin(), candidates.end(), farthest());

				nbh_v.resize(m_s);
				for(size_t k=0; k<m_s; ++k)
					nbh_v[k] = candidates[k].u;
				/*
				auto res = select_neighbors(get_node(pv).data, candidates, m_s, l_c);
				nbh_v.clear();
				for(auto *pu : res)
					nbh_v.push_back(pu);
				*/
				// nbh_v = select_neighbors(get_node(pv).data, candidates, m_s, l_c);
			}
			else nbh_v.insert(nbh_v.end(),nbh_v_add.begin(), nbh_v_add.end());
		});
	}

	debug_output("Updating entrance\n");
	// finally, update the entrance
	node_id node_highest = *std::max_element(
		node_new.get(), node_new.get()+size_batch, [&](const node_id u, const node_id v){
			return get_node(u).level < get_node(v).level;
	});
	if(get_node(node_highest).level>level_ep)
	{
		entrance.clear();
		entrance.push_back(node_highest);
		debug_output("New entrance [%u] at lev %u\n", U::get_id(get_node(node_highest).data), get_node(node_highest).level);
	}
	else if(get_node(node_highest).level==level_ep)
	{
		entrance.push_back(node_highest);
		debug_output("New entrance [%u] at lev %u\n", U::get_id(get_node(node_highest).data), get_node(node_highest).level);
	}

	// and add new nodes to the pool
	/*
	if(from_blank)
	node_pool.insert(node_pool.end(), node_new.get(), node_new.get()+size_batch);
	*/
}

template<class Conn, class G, class D, class Seq>
auto beamSearch(
	const G &g, D f_dist, const Seq &eps, uint32_t ef, const search_control &ctrl={})
{
	using nid_t = typename G::nid_t;
	using conn = Conn;

	const auto n = g.num_nodes();
	const uint32_t bits = ef>2? std::ceil(std::log2(ef))*2-2: 2;
	const uint32_t mask = (1u<<bits)-1;
	Seq visited(mask+1, n+1);
	uint32_t cnt_visited = 0;
	parlay::sequence<conn> workset;
	std::set<conn> cand; // TODO: test dual heaps
	std::unordered_set<nid_t> is_inw; // TODO: test merge instead
	// TODO: get statistics about the merged size
	// TODO: switch to the alternative if exceeding a threshold
	workset.reserve(ef+1);

	// debug_output("look at eps\n");
	for(nid_t pe : eps)
	{
		visited[parlay::hash64(pe)&mask] = pe;
		const auto d = f_dist(g.get_node(pe).data);
		cand.insert({d,pe});
		workset.push_back({d,pe});
		is_inw.insert(pe);
		cnt_visited++;
	}
	std::make_heap(workset.begin(), workset.end());

	uint32_t cnt_eval = 0;
	uint32_t limit_eval = ctrl.limit_eval.value_or(n);
	while(cand.size()>0)
	{
		if(cand.begin()->d>workset[0].d*ctrl.beta) break;

		if(++cnt_eval>limit_eval) break;

		nid_t u = cand.begin()->u;
		cand.erase(cand.begin());
		for(nid_t pv: g.get_edges(u))
		{
			const auto h_pv = parlay::hash64_2(pv)&mask;
			if(visited[h_pv]==pv) continue;
			visited[h_pv] = pv;
			cnt_visited++;

			const auto d = f_dist(g.get_node(pv).data);
			if(!(workset.size()<ef||d<workset[0].d)) continue;
			if(!is_inw.insert(pv).second) continue;

			cand.insert({d,pv});
			workset.push_back({d,pv});
			std::push_heap(workset.begin(), workset.end());
			if(workset.size()>ef)
			{
				std::pop_heap(workset.begin(), workset.end());
				// is_inw.erase(workset.back().u);
				workset.pop_back();
			}
			if(cand.size()>ef)
				cand.erase(std::prev(cand.end()));
		}
	}

	if(ctrl.count_cmps)
		*ctrl.count_cmps.value() += cnt_visited;

	return workset;
}

template<typename U, template<typename> class Allocator>
auto HNSW<U,Allocator>::search_layer(const node &u, const parlay::sequence<node_id> &eps, uint32_t ef, uint32_t l_c, search_control ctrl) const
{
	graph g(*this,l_c);
	/*
	dist_evaluator f_dist(u.data,dim);
	return beamSearch<dist>(g, f_dist, eps, ef, ctrl);
	*/
	QueryParams QP(ef, ef, 1.35, ctrl.limit_eval.value_or(n), get_threshold_m(l_c));
	auto points = parlay::delayed_seq<const T&>(node_pool.size(), [&](size_t i) -> const T&{
		return node_pool[i].data;
	});
	auto res = beam_search_impl<node_id>(u.data, g, points, eps, QP);
	const auto &pairElts = std::get<0>(res);
	const auto &frontier = std::get<0>(pairElts);
	if(ctrl.count_cmps)
		*ctrl.count_cmps.value() += std::get<1>(res);
	return parlay::tabulate(frontier.size(), [&](size_t i){
		const auto &f = frontier[i];
		return dist{f.second, f.first};
	});
}

template<typename U, template<typename> class Allocator>
auto HNSW<U,Allocator>::search_layer_bak(const node &u, const parlay::sequence<node_id> &eps, uint32_t ef, uint32_t l_c, search_control ctrl) const
{
	#define USE_HASHTBL
	// #define USE_BOOLARRAY
	// #define USE_UNORDERED_SET
#ifdef USE_HASHTBL
	const uint32_t bits = ef>2? std::ceil(std::log2(ef*ef))-2: 2;
	const uint32_t mask = (1u<<bits)-1;
	parlay::sequence<uint32_t> visited(mask+1, n+1);
#endif
#ifdef USE_BOOLARRAY
	std::vector<bool> visited(n+1);
#endif
	// TODO: Try hash to an array
	// TODO: monitor the size of `visited`
	uint32_t cnt_visited = 0;
#ifdef USE_UNORDERED_SET
	std::unordered_set<uint32_t> visited;
#endif
	parlay::sequence<dist> W, discarded;
	std::set<dist,farthest> C;
	std::set<node_id> w_inserted;
	W.reserve(ef+1);

	for(node_id ep : eps)
	{
	#ifdef USE_HASHTBL
		const auto id = U::get_id(get_node(ep).data);
		visited[parlay::hash64_2(id)&mask] = id;
	#endif
	#ifdef USE_BOOLARRAY
		visited[id] = true;
	#endif
	#ifdef USE_UNORDERED_SET
		visited.insert(U::get_id(get_node(ep).data));
	#endif
		cnt_visited++;
		const auto d = U::distance(u.data,get_node(ep).data,dim);
		C.insert({d,ep});
		W.push_back({d,ep});
		w_inserted.insert(ep);
	}
	// std::make_heap(C.begin(), C.end(), nearest());
	std::make_heap(W.begin(), W.end(), farthest());

	uint32_t cnt_eval = 0;
	uint32_t limit_eval = ctrl.limit_eval.value_or(n);
	while(C.size()>0)
	{
		if(ctrl.skip_search) break;
		if(C.begin()->d>W[0].d*ctrl.beta) break;

		if(++cnt_eval>limit_eval) break;
		if(ctrl.log_dist)
		{
			std::array<float,5> t;

			if(ctrl.log_size)
			{
				t[0] = W[0].d;
				t[1] = W.size();
				t[2] = C.size();
				vc_in_search[*ctrl.log_size].push_back(t);
			}

			auto it = C.begin();
			const auto step = C.size()/4;
			for(uint32_t i=0; i<4; ++i)
				t[i]=it->d, std::advance(it,step);
			t[4] = C.rbegin()->d;

			dist_in_search[*ctrl.log_dist].push_back(t);
		}

		const auto &c = get_node(C.begin()->u);
		// std::pop_heap(C.begin(), C.end(), nearest());
		// C.pop_back();
		C.erase(C.begin());
		for(node_id pv: neighbourhood(c, l_c))
		{
		#ifdef USE_HASHTBL
			const auto id = U::get_id(get_node(pv).data);
			const auto idx = parlay::hash64_2(id)&mask;
			if(visited[idx]==id) continue;
			visited[idx] = id;
		#endif
		#ifdef USE_BOOLARRAY
			if(visited[id]) continue;
			visited[id] = true;
		#endif
		#ifdef USE_UNORDERED_SET
			if(!visited.insert(U::get_id(get_node(pv).data)).second) continue;
		#endif
			cnt_visited++;
			const auto d = U::distance(u.data,get_node(pv).data,dim);
			if((W.size()<ef||d<W[0].d) && w_inserted.insert(pv).second)
			{
				C.insert({d,pv});
				// C.push_back({d,pv,dc+1});
				// std::push_heap(C.begin(), C.end(), nearest());
				W.push_back({d,pv});
				std::push_heap(W.begin(), W.end(), farthest());
				if(W.size()>ef)
				{
					std::pop_heap(W.begin(), W.end(), farthest());
					// w_inserted.erase(W.back().u);
					if(ctrl.radius && W.back().d<=*ctrl.radius)
						discarded.push_back(W.back());
					W.pop_back();
				}
				if(C.size()>ef)
					C.erase(std::prev(C.end()));
			}
		}
	}

	//total_visited += visited.size();
	//total_visited += visited.size()-std::count(visited.begin(),visited.end(),n+1);
	const auto id = parlay::worker_id();
	total_visited[id] += cnt_visited;
	total_size_C[id] += C.size()+cnt_eval;
	total_eval[id] += cnt_eval;

	if(ctrl.count_cmps)
		*ctrl.count_cmps.value() += cnt_visited;

	if(ctrl.radius)
	{
		const auto rad = *ctrl.radius;
		auto split = std::partition(W.begin(), W.end(), [rad](const dist &e){
			return e.d <= rad;
		});
		W.resize(split-W.begin());
		W.append(discarded);
		total_range_candidate[parlay::worker_id()] += W.size();
	}
	return W;
}

template<typename U, template<typename> class Allocator>
auto HNSW<U,Allocator>::search_layer_new_ex(const node &u, const parlay::sequence<node_id> &eps, uint32_t ef, uint32_t l_c, search_control ctrl) const
{
	auto verbose_output = [&](const char *fmt, ...){
		if(!ctrl.verbose_output) return;

		va_list args;
		va_start(args, fmt);
		vfprintf(stderr, fmt, args);
		va_end(args);
	};

	parlay::sequence<std::array<float,5>> dummy;
	auto &dist_range = ctrl.log_dist? dist_in_search[*ctrl.log_dist]: dummy;
	uint32_t cnt_eval = 0;

	auto *indeg = ctrl.verbose_output? get_indeg(l_c): reinterpret_cast<const uint32_t*>(node_pool.data());
	// parlay::sequence<bool> visited(n);
	// TODO: Try hash to an array
	// TODO: monitor the size of `visited`
	std::set<uint32_t> visited;
	// std::priority_queue<dist_ex,parlay::sequence<dist_ex>,nearest> C;
	// std::priority_queue<dist_ex,parlay::sequence<dist_ex>,farthest> W;
	parlay::sequence<dist_ex> /*C, W, */W_;
	std::set<dist_ex,farthest> C, C_acc;
	uint32_t cnt_used = 0;

	for(node_id ep : eps)
	{
		// visited[U::get_id(get_node(ep).data)] = true;
		const auto id = U::get_id(get_node(ep).data);
		visited.insert(id);
		const auto d = U::distance(u.data,get_node(ep).data,dim);
		C.insert({d,ep,1});
		C_acc.insert({d,ep,1});
		// C.push_back({d,ep,1});
		// W.push_back({d,ep,1});
		verbose_output("Insert\t[%u](%f) initially\n", id, d);
	}
	// std::make_heap(C.begin(), C.end(), nearest());
	// std::make_heap(W.begin(), W.end(), farthest());

	// static thread_local std::mt19937 gen{parlay::worker_id()};
	// static thread_local std::exponential_distribution<float> distro{48};
	while(C.size()>0)
	{
		// const auto &f = *(W[0].u);
		// if(U::distance(c.data,u.data,dim)>U::distance(f.data,u.data,dim))
		// if(C[0].d>W[0].d) break;
		if(C_acc.size()==cnt_used) break;
		cnt_eval++;

		if(ctrl.log_dist)
			dist_range.push_back({C.begin()->d,C.rbegin()->d});
		/*
		const auto dc = C[0].depth;
		const auto &c = *(C[0].u);
		*/
		auto it = C.begin();
		/*
		float quantile = distro(gen);
		if(quantile>C.size())
			quantile = C.size();
		const auto dis_min = C.begin()->d;
		const auto dis_max = C.rbegin()->d;
		const auto threshold = quantile/C.size()*(dis_max-dis_min) + dis_min - 1e-6;
		auto it = C.lower_bound(dist_ex{threshold,nullptr,0});
		*/
		const auto dc = it->depth;
		const auto &c = *(it->u);
		// W_.push_back(C[0]);
		W_.push_back(*it);
		// std::pop_heap(C.begin(), C.end(), nearest());
		// C.pop_back();
		C.erase(it);
		cnt_used++;

		verbose_output("------------------------------------\n");
		const uint32_t id_c = U::get_id(c.data);
		verbose_output("Eval\t[%u](%f){%u}\t[%u]\n", id_c, it->d, dc, indeg[id_c]);
		uint32_t cnt_insert = 0;
		for(node_id pv: neighbourhood(c, l_c))
		{
			// if(visited[U::get_id(get_node(pv).data)]) continue;
			// visited[U::get_id(get_node(pv).data)] = true;
			if(!visited.insert(U::get_id(get_node(pv).data)).second) continue;
			// const auto &f = *(W[0].u);
			// if(W.size()<ef||U::distance(get_node(pv).data,u.data,dim)<U::distance(f.data,u.data,dim))
			const auto d = U::distance(u.data,get_node(pv).data,dim);
			// if(W.size()<ef||d<W[0].d)
			// if(C.size()<ef||d<C.rend()->d)
			{
				// C.push_back({d,pv,dc+1});
				// std::push_heap(C.begin(), C.end(), nearest());
				/*
				W.push_back({d,pv,dc+1});
				std::push_heap(W.begin(), W.end(), farthest());
				if(W.size()>ef)
				{
					std::pop_heap(W.begin(), W.end(), farthest());
					W.pop_back();
				}
				*/
				if(C.size()<ef || d<C.rbegin()->d)
				{
				C.insert({d,pv,dc+1});
				const uint32_t id_v = U::get_id(get_node(pv).data);
				verbose_output("Insert\t[%u](%f){%u}\t[%u](%f)\n", 
					id_v, d, dc+1, 
					indeg[id_v], U::distance(c.data,get_node(pv).data,dim)
				);
				cnt_insert++;
				if(C.size()>ef)
				{
					// std::pop_heap(C.begin(), C.end(), nearest());
					// C.pop_back();
					C.erase(std::prev(C.end()));
				}
				}
				if(C_acc.size()<ef || d<C_acc.rbegin()->d)
				{
				C_acc.insert({d,pv,dc+1});
				if(C_acc.size()>ef)
				{
					auto it = std::prev(C_acc.end());
					if(std::find_if(W_.begin(), W_.end(), [&](const dist_ex &a){
						return a.u==it->u;
					})!=W_.end())
						cnt_used--;
					C_acc.erase(it);
				}
				}
			}
		}
		verbose_output("%u inserts in this round\n", cnt_insert);
	}
	if(l_c==0)
	{
		const auto id = parlay::worker_id();
		total_visited[id] += visited.size();
		total_size_C[id] += C.size()+cnt_eval;
		total_eval[id] += cnt_eval;
	}
	/*
	std::sort(W.begin(), W.end(), farthest());
	if(W.size()>ef) W.resize(ef);
	*/
	return W_;
}

template<typename U, template<typename> class Allocator>
auto HNSW<U,Allocator>::beam_search_ex(const node &u, const parlay::sequence<node_id> &eps, uint32_t beamSize, uint32_t l_c, search_control ctrl) const
// std::pair<parlay::sequence<dist_ex>, parlay::sequence<dist_ex>> beam_search(
		// T* p_coords, int beamSize)
{
	// beamSize *= 2;
	// beamSize = 20000;
	// initialize data structures
	parlay::sequence<dist_ex> visited;
	parlay::sequence<dist_ex> frontier;
	auto dist_less = [&](const dist_ex &a, const dist_ex &b) {
		return a.d < b.d || (a.d == b.d && a.u < b.u);
		// return a.u<b.u;
	};
	auto dist_eq = [&](const dist_ex &a, const dist_ex &b){
		return a.u == b.u;
	};

	// int bits = std::ceil(std::log2(beamSize * beamSize));
	// parlay::sequence<uint32_t> hash_table(1 << bits, std::numeric_limits<uint32_t>::max());
	std::set<uint32_t> accessed;

	auto make_pid = [&] (node_id ep) {
		const auto d = U::distance(u.data,get_node(ep).data,dim);
		return dist_ex{d,ep,1};
	};

	// the frontier starts with the medoid
	// frontier.push_back(make_pid(medoid->id));
	
	for(node_id ep : eps)
		frontier.push_back(make_pid(ep));
	std::sort(frontier.begin(), frontier.end(), dist_less);
	
	// frontier.push_back(make_pid(eps[0]));

	parlay::sequence<dist_ex> unvisited_frontier;
	// parlay::sequence<dist_ex> unvisited_frontier(beamSize);
	parlay::sequence<dist_ex> new_frontier;
	// parlay::sequence<dist_ex> new_frontier(2 * beamSize);
	bool not_done = true;


	for(size_t i=0; i<frontier.size(); ++i)
	{
		unvisited_frontier.push_back(frontier[i]);
		// unvisited_frontier[i] = frontier[i];
		accessed.insert(U::get_id(frontier[i].get_node(u).data));
	}

	// terminate beam search when the entire frontier has been visited
	while (not_done) {
		// the next node to visit is the unvisited frontier node that is closest
		// to p
		dist_ex currentPid = unvisited_frontier[0];
		node_id current_vtx = currentPid.u;
		debug_output("current_vtx ID: %u\n", U::get_id(get_node(current_vtx).data));

		auto g = [&](node_id a) {
			uint32_t id_a = U::get_id(get_node(a).data);
			/*
			uint32_t loc = parlay::hash64_2(id_a) & ((1 << bits) - 1);
			if (hash_table[loc] == id_a) return false;
			hash_table[loc] = id_a;
			return true;
			*/
			return accessed.insert(id_a).second;
		};

		parlay::sequence<node_id> candidates;
		auto f = [&](node_id pu, node_id pv/*, empty_weight wgh*/) {
			if (g(pv)) {
				candidates.push_back(pv);
			}
			return true;
		};
		for(node_id pv : neighbourhood(get_node(current_vtx),l_c))
			// current_vtx.out_neighbors().foreach_cond(f);
			f(current_vtx, pv);

		debug_output("candidates:\n");
		for(node_id p : candidates)
			debug_output("%u ", U::get_id(get_node(p).data));
		debug_output("\n");
		auto pairCandidates =
				parlay::map(candidates, make_pid);
		/*
		auto sortedCandidates =
				parlay::unique(parlay::sort(pairCandidates, dist_less), dist_eq);
		*/
		auto &sortedCandidates = pairCandidates;
		debug_output("size of sortedCandidates: %lu\n", sortedCandidates.size());
		/*
		auto f_iter = std::set_union(
				frontier.begin(), frontier.end(), sortedCandidates.begin(),
				sortedCandidates.end(), new_frontier.begin(), dist_less);\
		*/
		sortedCandidates.insert(sortedCandidates.end(), frontier);
		new_frontier = parlay::unique(parlay::sort(sortedCandidates,dist_less), dist_eq);

		// size_t f_size = std::min<size_t>(beamSize, f_iter - new_frontier.begin());
		size_t f_size = std::min<size_t>(beamSize, new_frontier.size());
		debug_output("f_size: %lu\n", f_size);

		debug_output("frontier (size: %lu)\n", frontier.size());
		for(const auto &e : frontier)
			debug_output("%u ", U::get_id(e.get_node(u).data));
		debug_output("\n");
		
		frontier =
				parlay::tabulate(f_size, [&](size_t i) { return new_frontier[i]; });
		debug_output("size of frontier: %lu\n", frontier.size());
		visited.insert(
				std::upper_bound(visited.begin(), visited.end(), currentPid, dist_less),
				currentPid);
		debug_output("size of visited: %lu\n", visited.size());
		unvisited_frontier.reserve(frontier.size());
		auto uf_iter =
				std::set_difference(frontier.begin(), frontier.end(), visited.begin(),
														visited.end(), unvisited_frontier.begin(), dist_less);
		debug_output("uf_iter - unvisited_frontier.begin(): %lu\n", uf_iter - unvisited_frontier.begin());
		not_done = uf_iter > unvisited_frontier.begin();

		if(l_c==0)
			total_visited[parlay::worker_id()] += candidates.size();
	}
	parlay::sequence<dist_ex> W;
	W.insert(W.end(), visited);
	return W;
}
/*
template<typename U, template<typename> class Allocator>
parlay::sequence<std::pair<uint32_t,float>> HNSW<U,Allocator>::search(const T &q, uint32_t k, uint32_t ef, search_control ctrl)
{
	auto res_ex = search_ex(q,k,ef,ctrl);
	parlay::sequence<std::pair<uint32_t,float>> res;
	res.reserve(res_ex.size());
	for(const auto &e : res_ex)
		res.emplace_back(std::get<0>(e), std::get<2>(e));

	return res;
}
*/

template<typename U, template<typename> class Allocator>
parlay::sequence<typename HNSW<U,Allocator>::node_id> HNSW<U,Allocator>::search_layer_to(
	const node &u, uint32_t ef, uint32_t l_stop, const search_control &ctrl)
{
	auto eps = entrance;
	for(uint32_t l_c=get_node(entrance[0]).level; l_c>l_stop; --l_c)
	{
		search_control c{};
		c.log_per_stat = ctrl.log_per_stat; // whether count dist calculations at all layers
		// c.limit_eval = ctrl.limit_eval; // whether apply the limit to all layers
		c.count_cmps = ctrl.count_cmps;
		const auto W = search_layer(u, eps, ef, l_c, c);
		eps.clear();
		eps.push_back(W[0].u);
		/*
		while(!W.empty())
		{
			eps.push_back(W.top().u);
			W.pop();
		}
		*/
	}
	return eps;
}

template<typename U, template<typename> class Allocator>
parlay::sequence<std::pair<uint32_t,float>> HNSW<U,Allocator>::search(
	const T &q, uint32_t k, uint32_t ef, const search_control &ctrl)
{
	const auto id = parlay::worker_id();
	total_range_candidate[id] = 0;
	total_visited[id] = 0;
	total_eval[id] = 0;
	total_size_C[id] = 0;

	node u{n, nullptr, q}; // To optimize
	// std::priority_queue<dist,parlay::sequence<dist>,farthest> W;
	parlay::sequence<node_id> eps;
	if(ctrl.indicate_ep)
		eps.push_back(*ctrl.indicate_ep);
	else
		eps = search_layer_to(u, 1, 0, ctrl);
	auto W_ex = search_layer(u, eps, ef, 0, ctrl);
	// auto W_ex = search_layer_new_ex(u, eps, ef, 0, ctrl);
	// auto W_ex = beam_search_ex(u, eps, ef, 0);
	// auto R = select_neighbors_simple(q, W_ex, k);

	auto &R = W_ex;
	if(!ctrl.radius && R.size()>k) // the range search ignores the given k
	{
		std::sort(R.begin(), R.end(), farthest());
		if(k>0)
			k = std::upper_bound(R.begin()+k, R.end(), R[k-1], farthest())-R.begin();
		R.resize(k);
	}

	parlay::sequence<std::pair<uint32_t,float>> res;
	res.reserve(R.size());
	/*
	while(W_ex.size()>0)
	{
		res.push_back({U::get_id(W_ex.top().get_node(u).data), W_ex.top().depth, W_ex.top().d});
		W_ex.pop();
	}
	*/
	for(const auto &e : R)
		res.push_back({U::get_id(get_node(e.u).data),/* e.depth,*/ e.d});
	return res;
}

template<typename U, template<typename> class Allocator>
void HNSW<U,Allocator>::save(const std::string &filename_model) const
{
	std::ofstream model(filename_model, std::ios::binary);
	if(!model.is_open())
		throw std::runtime_error("Failed to create the model");

	const auto size_buffer = 1024*1024*1024; // 1G
	auto buffer = std::make_unique<char[]>(size_buffer);
	model.rdbuf()->pubsetbuf(buffer.get(), size_buffer);

	const auto write = [&](const auto &data, auto ...args){
		auto write_impl = [&](auto &f, const auto &data, auto ...args){
			using T = std::remove_reference_t<decltype(data)>;
			if constexpr(std::is_pointer_v<std::decay_t<T>>)
			{
				auto write_array = [&](const auto &data, size_t size, auto ...args){
					for(size_t i=0; i<size; ++i)
						f(f, data[i], args...);
				};
				// use the array extent as the size
				if constexpr(sizeof...(args)==0 && std::is_array_v<T>)
				{
					write_array(data, std::extent_v<T>);
				}
				else
				{
					static_assert(sizeof...(args), "size was not provided");
					write_array(data, args...);
				}
			}
			else
			{
				static_assert(std::is_standard_layout_v<T>);
				model.write((const char*)&data, sizeof(data));
			}
		};
		write_impl(write_impl, data, args...);
	};
	// write header (version number, type info, etc)
	write("HNSW", 4);
	write(uint32_t(3)); // version
	write(typeid(U).hash_code()^sizeof(U));
	fprintf(stderr, "U type written %s\n", typeid(U).name());
	write(sizeof(node));
	// write parameter configuration
	write(dim);
	write(m_l);
	write(m);
	write(ef_construction);
	write(alpha);
	write(n);
	// write indices
	for(const auto &u : node_pool)
	{
		write(u.level);
		write(uint32_t(U::get_id(u.data)));
	}
	for(const auto &u : node_pool)
	{
		for(uint32_t l=0; l<=u.level; ++l)
		{
			write(u.neighbors[l].size());
			for(node_id pv : u.neighbors[l])
				write(pv);
		}
	}
	// write entrances
	write(entrance.size());
	for(node_id pu : entrance)
		write(pu);
} 

} // namespace HNSW

#endif // _HNSW_HPP

