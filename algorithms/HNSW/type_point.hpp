#ifndef __TYPE_POINT_HPP__
#define __TYPE_POINT_HPP__

#include <cstdint>
#include <cstddef>
#include <iterator>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <stdexcept>
#include <any>
#include "benchUtils.h"

#ifdef SUPPORT_HDF5
#include "h5_ops.hpp"
#endif

class internal_termination{
protected:
	internal_termination(){}
	internal_termination(int){std::terminate();}
};

template<typename T>
class fake_copyable : public internal_termination{
	T content;
public:
	fake_copyable(const T &c) : content(c){}
	fake_copyable(T &&c) : content(std::move(c)){}

	fake_copyable(fake_copyable&&) = default;
	fake_copyable(const fake_copyable &other [[maybe_unused]])
	// The users have to guarantee to hold the points while it is being used in graph.
	// Otherwise, uncomment the following guarding code and forbid copy constructions
	// or alternatively pass in copy-constructible objects (e.g., `std::shared_ptr`) 
	// to `point` instead of using this hack
	/*
		: internal_termination(0), 
		  content(std::move(const_cast<fake_copyable&>(other).content))
	*/
		: internal_termination()
	{
	}
};
template<typename T>
fake_copyable(const T&) -> fake_copyable<T>;
template<typename T>
fake_copyable(T&&) -> fake_copyable<T>;

template<typename T>
struct point
{
	typedef T type;

	uint32_t id;
	const T *coord;

	point()
		: id(~0u), coord(NULL), closure()
	{
	}
	point(uint32_t id_, const T *coord_)
		: id(id_), coord(coord_), closure()
	{
	}
	template<class C>
	point(uint32_t id_, const T *coord_, C &&closure_)
		: id(id_), coord(coord_), closure(std::forward<C>(closure_))
	{
	}
private:
	std::any closure;
};

enum class file_format{
	VEC, HDF5, BIN
};

template<typename T>
class point_converter_default
{
public:
	using type = point<T>;

	template<typename Iter>
	type operator()(uint32_t id, Iter begin, [[maybe_unused]] Iter end)
	{
		using type_src = typename std::iterator_traits<Iter>::value_type;
		static_assert(std::is_convertible_v<type_src,T>, "Cannot convert to the target type");

		if constexpr(std::is_same_v<Iter,ptr_mapped<T,ptr_mapped_src::PERSISTENT>>||
			std::is_same_v<Iter,ptr_mapped<const T,ptr_mapped_src::PERSISTENT>>)
			return point<T>(id, &*begin);
		else if constexpr(std::is_same_v<Iter,ptr_mapped<T,ptr_mapped_src::TRANSITIVE>>||
			std::is_same_v<Iter,ptr_mapped<const T,ptr_mapped_src::TRANSITIVE>>)
		{
			const T *p = &*begin; // TODO: fix the type to T(*)[]
			return point<T>(id, p, fake_copyable(std::unique_ptr<const T>(p)));
		}
		else
		{
			const uint32_t dim = std::distance(begin, end);

			// T *coord = new T[dim];
			auto coord = std::make_unique<T[]>(dim);
			for(uint32_t i=0; i<dim; ++i)
				coord[i] = *(begin+i);
			return point<T>(id, coord.get(), fake_copyable(std::move(coord)));
		}
	}
};

template<typename Src, class Conv>
inline std::pair<parlay::sequence<typename Conv::type>,uint32_t>
load_from_vec(const char *file, Conv converter, uint32_t max_num)
{
	const auto [fileptr, length] = mmapStringFromFile(file);

	// Each vector is 4 + sizeof(Src)*dim bytes.
	// * first 4 bytes encode the dimension (as an uint32_t)
	// * next dim values are Src-type variables representing vector components
	// See http://corpus-texmex.irisa.fr/ for more details.

	const uint32_t dim = *((const uint32_t*)fileptr);
	std::cout << "Dimension = " << dim << std::endl;

	const size_t vector_size = sizeof(dim) + sizeof(Src)*dim;
	const uint32_t n = std::min<size_t>(length/vector_size, max_num);
	// std::cout << "Num vectors = " << n << std::endl;

	typedef ptr_mapped<const Src,ptr_mapped_src::PERSISTENT> type_ptr;
	parlay::sequence<typename Conv::type> ps(n);

	parlay::parallel_for(0, n, [&,fp=fileptr] (size_t i) {
		const Src *coord = (const Src*)(fp+sizeof(dim)+i*vector_size);
		ps[i] = converter(i, type_ptr(coord), type_ptr(coord+dim));
	});

	return {std::move(ps), dim};
}

template<class, class=void>
class trait_type{
};

template<class T>
class trait_type<T,std::void_t<typename T::type>>{
public:
	using type = typename T::type;
};

template<class T>
class trait_type<T*,void>{
public:
	using type = T;
};

template<class T>
class trait_type<parlay::sequence<T>,void>{
public:
	using type = T;
};

template<class Conv>
inline std::pair<parlay::sequence<typename Conv::type>,uint32_t>
load_from_HDF5(const char *file, const char *dir, Conv converter, uint32_t max_num)
{
#ifndef SUPPORT_HDF5
	(void)file;
	(void)dir;
	(void)converter;
	(void)max_num;
	throw std::invalid_argument("HDF5 support is not enabled");
#else
	using T = typename trait_type<typename Conv::type>::type;
	auto [reader,bound] = get_reader<T>(file, dir);
	const size_t n = std::min<size_t>(bound[0], max_num);
	const uint32_t dim = bound[1];

	parlay::sequence<typename Conv::type> ps(n);
	// TODO: parallel for-loop
	for(uint32_t i=0; i<n; ++i){
		T *coord = new T[dim];
		reader(coord, i);
		typedef ptr_mapped<T,ptr_mapped_src::TRANSITIVE> type_ptr;
		ps[i] = converter(i, type_ptr(coord), type_ptr(coord+dim));
	}
	return {std::move(ps), dim};
#endif
}

template<typename Src, class Conv>
inline std::pair<parlay::sequence<typename Conv::type>,uint32_t>
load_from_bin(const char *file, Conv converter, uint32_t max_num)
{
	auto [fileptr, length] = mmapStringFromFile(file); (void)length;
	const uint32_t n = std::min(max_num, *((uint32_t*)fileptr));
	const uint32_t dim = *((uint32_t*)(fileptr+sizeof(n)));
	const size_t vector_size = sizeof(Src)*dim;
	const size_t header_size = sizeof(n)+sizeof(dim);

	typedef ptr_mapped<const Src,ptr_mapped_src::PERSISTENT> type_ptr;
	parlay::sequence<typename Conv::type> ps(n);
	parlay::parallel_for(0, n, [&,fp=fileptr](uint32_t i){
		const Src *coord = (const Src*)(fp+header_size+i*vector_size);
		ps[i] = converter(i, type_ptr(coord), type_ptr(coord+dim));
	});

	return {std::move(ps), dim};
}

template<typename Src, class Conv>
inline std::pair<parlay::sequence<typename Conv::type>,uint32_t>
load_from_range(const char *file, Conv converter, uint32_t max_num)
{
	auto [fileptr, length] = mmapStringFromFile(file); (void)length;
	const int32_t num_points = *(int32_t*)fileptr;
	const int32_t num_matches = *(int32_t*)(fileptr+sizeof(num_points));
	const size_t header_size = sizeof(num_points)+sizeof(num_matches);

	int32_t* begin = (int32_t*)(fileptr+header_size);
	int32_t* end = begin + num_points;
	auto [offsets, total] = parlay::scan(parlay::make_slice(begin,end));
	offsets.push_back(total);
	std::cout << "num_matches: " << num_matches << ' ' << total << std::endl;

	const size_t index_size = header_size+num_points*sizeof(*begin);
	std::cout << "index_size: " << index_size << std::endl;

	typedef ptr_mapped<const Src,ptr_mapped_src::PERSISTENT> type_ptr;
	const uint32_t n = std::min<uint32_t>(max_num, num_points);
	parlay::sequence<typename Conv::type> ps(n);
	parlay::parallel_for(0, n, [&,fp=fileptr](uint32_t i){
		const Src *begin = (const Src*)(fp+index_size+offsets[i]*sizeof(Src));
		const Src *end = (const Src*)(fp+index_size+offsets[i+1]*sizeof(Src));
		ps[i] = converter(i, type_ptr(begin), type_ptr(end));
	});

	return {std::move(ps), 0};
}
/*
template<typename Src=void, class Conv>
inline auto load_point(const char *file, file_format input_format, Conv converter, size_t max_num=0, std::any aux={})
{
	if(!max_num)
		max_num = std::numeric_limits<decltype(max_num)>::max();

	switch(input_format)
	{
	case file_format::VEC:
		return load_from_vec<Src>(file, converter, max_num);
	case file_format::HDF5:
		return load_from_HDF5(file, std::any_cast<const char*>(aux), converter, max_num);
	case file_format::BIN:
		return load_from_bin<Src>(file, converter, max_num);
	default:
		__builtin_unreachable();
	}
}
*/
template<class Conv>
inline auto load_point(const char *input_name, Conv converter, size_t max_num=0)
{
	auto buffer = std::make_unique<char[]>(strlen(input_name)+1);
	strcpy(buffer.get(), input_name);

	char *splitter = strchr(buffer.get(), ':');
	if(splitter==nullptr)
		throw std::invalid_argument("The input spec is not specified");

	*(splitter++) = '\0';
	const char *file = buffer.get();
	const char *input_spec = splitter;

	if(!max_num)
		max_num = std::numeric_limits<decltype(max_num)>::max();

	if(input_spec[0]=='/')
		return load_from_HDF5(file, input_spec, converter, max_num);
	if(!strcmp(input_spec,"fvecs"))
		return load_from_vec<float>(file, converter, max_num);
	if(!strcmp(input_spec,"bvecs"))
		return load_from_vec<uint8_t>(file, converter, max_num);
	if(!strcmp(input_spec,"ivecs"))
		return load_from_vec<int32_t>(file, converter, max_num);
	if(!strcmp(input_spec,"u8bin"))
		return load_from_bin<uint8_t>(file, converter, max_num);
	if(!strcmp(input_spec,"i8bin"))
		return load_from_bin<int8_t>(file, converter, max_num);
	if(!strcmp(input_spec,"ibin"))
		return load_from_bin<int32_t>(file, converter, max_num);
	if(!strcmp(input_spec,"ubin"))
		return load_from_bin<uint32_t>(file, converter, max_num);
	if(!strcmp(input_spec,"fbin"))
		return load_from_bin<float>(file, converter, max_num);
	if(!strcmp(input_spec,"irange"))
		return load_from_range<int32_t>(file, converter, max_num);

	throw std::invalid_argument("Unsupported input spec");
}

#endif // __TYPE_POINT_HPP_
