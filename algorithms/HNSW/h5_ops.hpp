#ifndef __H5_OPS_HPP__
#define __H5_OPS_HPP__

#include <cstdio>
#include <array>
#include <tuple>
#include <memory>
#include <type_traits>
#include <H5Cpp.h>

// Return a {reader, dims} tuple
// reader(buffer, index, cnt) reads data with the first dimension 
// from  `index` to `index+cnt` and writes it into the 1D `buffer`
template<typename T>
auto get_reader(const char *file, const char *dir)
{
	H5::H5File file_h5(file, H5F_ACC_RDONLY);
	H5::DataSet dset = file_h5.openDataSet(dir);
	H5::DataSpace dspace_src = dset.getSpace();
	hsize_t dim[2];
	dspace_src.getSimpleExtentDims(dim);
	fprintf(stderr, "%s: [%llu,%llu]\n", dir, dim[0], dim[1]);

	H5::DataType type_dst = dset.getDataType();
	if constexpr(std::is_same_v<T,uint32_t>)
		type_dst = H5::PredType::NATIVE_UINT32;
	else if constexpr(std::is_same_v<T,int32_t>)
		type_dst = H5::PredType::NATIVE_INT32;
	else if constexpr(std::is_same_v<T,uint8_t>)
		type_dst = H5::PredType::NATIVE_UINT8;
	else if constexpr(std::is_same_v<T,int8_t>)
		type_dst = H5::PredType::NATIVE_INT8;
	else if constexpr(std::is_same_v<T,float>)
		type_dst = H5::PredType::NATIVE_FLOAT;
	else static_assert(std::is_same_v<T,uint32_t>/*always false*/, "Unsupported type");

	auto reader = [=,_=std::move(file_h5)](T *buffer, hsize_t index, hsize_t cnt=1){
		hsize_t size = dim[1]*cnt;
		H5::DataSpace dspace_dst(1,&size,NULL);

		hsize_t offset[2] = {index, 0};
		hsize_t count[2] = {cnt, dim[1]};
		H5::DataSpace dspace_slice;
		dspace_slice.copy(dspace_src);
		dspace_slice.selectHyperslab(H5S_SELECT_SET, count, offset);

		dset.read(buffer, type_dst, dspace_dst, dspace_slice);
	};

	return std::tuple{reader, std::array{dim[0], dim[1]}};
}

// read a 2D array from H5 file and return a 1D array
template<typename T>
std::pair<std::unique_ptr<T[]>,std::array<hsize_t,2>> read_array_from_HDF5(const char *file, const char *dir)
{
	auto [reader,dims] = get_reader<T>(file, dir);
	auto buffer = std::make_unique<T[]>(dims[0]*dims[1]);
	reader(buffer.get(), 0, dims[0]);
	return {std::move(buffer), dims};
}

#endif // __H5_OPS_HPP__
