/* interface for point filters represented in CSR format

The fact that we use int32 for one of the indices and require transposes means we realistically are only going to be able to handle 2^31 points and 2^31 filters, but that's probably fine for now.
*/
#ifndef CSR_FILTER_H
#define CSR_FILTER_H

#include "parlay/sequence.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility>
#include <stdint.h>

/* Sorted array join (with pointers)*/
template <typename T>
parlay::sequence<T> join(const T* a, size_t len_a, const T* b, size_t len_b){
    parlay::sequence<T> output = parlay::sequence<T>();
    size_t i = 0;
    size_t j = 0;
    output.reserve(std::min(len_a, len_b));

    while(i < len_a && j < len_b){
        if(a[i] < b[j]){
            i++;
        } else if(a[i] > b[j]){
            j++;
        } else {
            output.push_back(a[i]);
            i++;
            j++;
        }
    }
    return std::move(output);
}

/* Sorted array join (with sequences) */
template <typename T>
inline parlay::sequence<T> join(const parlay::sequence<T>& a, const parlay::sequence<T>& b) {
    return join<T>(a.begin(), a.size(), b.begin(), b.size()); // hopefully this is virtually free
}


struct QueryFilter {
    int32_t a, b;

    QueryFilter(int32_t a, int32_t b) : a(a), b(b) {}

    QueryFilter(int32_t a) : a(a), b(-1) {}

    bool is_and() const {
        // because the only possible negative value is -1, we can return the inverse of the sign bit of b
        return ~b >> 31;
    }

    parlay::sequence<int32_t> get_sequence() const {
        if (is_and()) {
            return parlay::sequence<int32_t>({a, b});
        } else {
            return parlay::sequence<int32_t>({a});
        }
    }
};


struct csr_filters{
    int64_t n_points;
    int64_t n_filters;
    int64_t n_nonzero;
    std::unique_ptr<int64_t[]> row_offsets; // indices into data
    std::unique_ptr<int32_t[]> row_indices; // the indices of the nonzero entries, which is actually all we need
    bool transposed = false;

    csr_filters() = default;

    /* mmaps filter data in csr form from filename */
    csr_filters(std::string filename) {
        // opening file stream
        FILE* fp = fopen(filename.c_str(), "rb");
        if (fp == NULL) {
            fprintf(stderr, "Error opening file %s\n", filename);
            exit(1);
        }

        // reading in number of points, filters, and nonzeros
        fread(&n_points, sizeof(int64_t), 1, fp);
        fread(&n_filters, sizeof(int64_t), 1, fp);
        fread(&n_nonzero, sizeof(int64_t), 1, fp);

        // reading in row offsets
        row_offsets = std::make_unique<int64_t[]>(n_points + 1);
        fread(row_offsets.get(), sizeof(int64_t), n_points + 1, fp);

        // reading in row indices
        row_indices = std::make_unique<int32_t[]>(n_nonzero);
        fread(row_indices.get(), sizeof(int32_t), n_nonzero, fp);

        fclose(fp);

        for (int64_t i = 0; i < n_points; i++) {
            std::sort(row_indices.get() + row_offsets[i], row_indices.get() + row_offsets[i + 1]);
        }
    }

    // /* constructs csr_filters with all the values already provided as arguments (should probably be protected)*/
    // csr_filters(int64_t n_points, int64_t n_filters, int64_t n_nonzero, std::unique_ptr<int64_t[]> row_offsets, std::unique_ptr<int32_t[]> row_indices) : n_points(n_points), n_filters(n_filters), n_nonzero(n_nonzero), row_offsets(row_offsets), row_indices(row_indices) {
    //     std::cout << "first offset: " << row_offsets[0] << std::endl;
    //     std::cout << "first column: " << row_indices[0] << std::endl;
    // }

    csr_filters(const csr_filters& other) { // copy constructor
        // std::cout << "copying" << std::endl;
        this->n_points = other.n_points;
        this->n_filters = other.n_filters;
        this->n_nonzero = other.n_nonzero;

        // copying the dynamically allocated arrays
        row_offsets = std::make_unique<int64_t[]>(this->n_points + 1);
        row_indices = std::make_unique<int32_t[]>(this->n_nonzero);
        std::copy(other.row_offsets.get(), other.row_offsets.get() + n_points + 1, row_offsets.get());
        std::copy(other.row_indices.get(), other.row_indices.get() + n_nonzero, row_indices.get());

        this->transposed = other.transposed;
    }

    csr_filters& operator=(const csr_filters& other) { // copy assignment
        // std::cout << "copying from assignment" << std::endl;

        if (this == &other) { // self assignment
            return *this;
        }

        this->n_points = other.n_points;
        this->n_filters = other.n_filters;
        this->n_nonzero = other.n_nonzero;

        // copying the dynamically allocated arrays
        row_offsets = std::make_unique<int64_t[]>(this->n_points + 1);
        row_indices = std::make_unique<int32_t[]>(this->n_nonzero);
        std::copy(other.row_offsets.get(), other.row_offsets.get() + n_points + 1, row_offsets.get());
        std::copy(other.row_indices.get(), other.row_indices.get() + n_nonzero, row_indices.get());

        this->transposed = other.transposed;
        return *this;
    }


    ~csr_filters() = default;

    void print_stats() const {
        printf("n_points: %ld\n", n_points);
        printf("n_filters: %ld\n", n_filters);
        printf("n_nonzeros: %ld\n", n_nonzero);
    }

    /* Returns true if p matches filter f, which is equivalent to row p column f being nonzero

    Uses linear scan
    */
    bool match(int64_t p, int64_t f) const {
        // if (transposed) {
        //     std::swap(p, f);
        // }

        int64_t start = row_offsets[p];
        int64_t end = row_offsets[p + 1];

        // linear scan over row to see if f is in it, which should be fast since rows are short but vectorization or binary search could be worth it
        for (int64_t i = start; i < end; i++) {
            if (row_indices[i] == f) {
                return true;
            }
        }
        return false;
    }

    /* Returns true if p matches filter f, which is equivalent to row p column f being nonzero

    Uses binary search
    */
    bool bin_match(int64_t p, int64_t f) const {
        int64_t start =  row_offsets[p];
        int64_t end = row_offsets[p + 1];

        // binary search over row to see if f is in it
        while (start < end) {
            int64_t mid = (start + end) / 2;
            if (row_indices[mid] == f) {
                return true;
            } else if (row_indices[mid] < f) {
                start = mid + 1;
            } else {
                end = mid - 1;
            }
        }
        return false;
    }
    
    /* Returns true if p matches filter f, which is equivalent to row p column f being nonzero

    Uses std::find
    */
    bool std_match(int64_t p, int64_t f) const {
        int64_t start = row_offsets[p];
        int64_t end = row_offsets[p + 1];
        return std::find(row_indices.get() + start, row_indices.get() + end, f) != row_indices.get() + end;
    }

    /* returns indices of points matching QueryFilter
    */
    parlay::sequence<int32_t> query_matches(QueryFilter q) const {
        if (not transposed) {
            std::cout << "You are attempting to query a non-transposed csr_filter. This would require iterating over all the points in the dataset, which is almost certainly not what you want to do. Transpose this object." << std::endl;
            exit(1);
        };
        if (q.is_and()) {
            return join(row_indices.get() + row_offsets[q.a], row_offsets[q.a + 1] - row_offsets[q.a], row_indices.get() + row_offsets[q.b], row_offsets[q.b + 1] - row_offsets[q.b]);
        } else {
            return parlay::sequence<int32_t>(row_indices.get() + row_offsets[q.a], row_indices.get() + row_offsets[q.a + 1]);
        }
    }

    /* I would like to be able to get the filters associated with a point in a python-accesible way but this is a good enough proof of concept until I figure out how to do that */
    int64_t first_label(int64_t p) const {
        return row_indices[row_offsets[0]];
    }

    /* Returns the number of points matching a given filter */
    int64_t filter_count(int64_t f) const {
        return parlay::reduce(parlay::delayed_seq<int64_t>(n_points, [&] (int64_t i) {
            return match(i, f);
        }));
    }

    /* Returns the number of filters associated with a point */
    int64_t point_count(int64_t p) const {
        return row_offsets[p + 1] - row_offsets[p];
    }

    parlay::sequence<int64_t> filter_counts() const {
        return parlay::tabulate(n_filters, [&] (int64_t i) {
            return filter_count(i);
        });
    }

    /* Returns the indices of the filters associated with a point */
    parlay::sequence<int32_t> point_filters(int64_t p) const {
        return parlay::sequence<int32_t>(row_indices.get() + row_offsets[p], row_indices.get() + row_offsets[p + 1]);
    }

    /* Returns the intersection of the filters between two points */
    parlay::sequence<int32_t> point_intersection(int64_t a, int64_t b) const {
        return join(row_indices.get() + row_offsets[a], row_offsets[a + 1] - row_offsets[a], row_indices.get() + row_offsets[b], row_offsets[b + 1] - row_offsets[b]);
    }

    /* Transposes to make acessing points associated with a filter fast */
    csr_filters transpose() const {
        csr_filters out = *this;
        out.transpose_inplace();
        return out;
    }

    /* transposes the filters in place */
    void transpose_inplace() {
        // std::cout << "Transposing (inplace)..." << std::endl;

        std::unique_ptr<int64_t[]> new_row_offsets = std::make_unique<int64_t[]>(n_filters + 1);
        std::unique_ptr<int32_t[]> new_row_indices = std::make_unique<int32_t[]>(n_nonzero);

        memset(new_row_offsets.get(), 0, (n_filters + 1) * sizeof(int64_t)); // initializing to 0s

        // counting points associated with each filter and scanning to get row offsets
        for (int64_t i = 0; i < n_nonzero; i++) {
            new_row_offsets[row_indices[i] + 1]++;
        }

        // not a sequence so for now I'll just do it serially
        for (int64_t i = 1; i < n_filters + 1; i++) {
            new_row_offsets[i] += new_row_offsets[i - 1];
        }
        // std::cout << "Offsets computed" << std::endl;

        // int64_t* tmp_offset = (int64_t*) malloc(n_filters * sizeof(int64_t)); // temporary array to keep track of where to put the next point in each filter
        std::unique_ptr<int64_t[]> tmp_offset = std::make_unique<int64_t[]>(n_filters);
        memset(tmp_offset.get(), 0, n_filters * sizeof(int64_t)); // initializing to 0s

        // iterating over the data to fill in row indices
        for (int64_t i = 0; i < n_points; i++) {
            int64_t start = row_offsets[i];
            int64_t end = row_offsets[i + 1];
            for (int64_t j = start; j < end; j++) {
                int64_t f = row_indices[j];
                int64_t index = new_row_offsets[f] + tmp_offset[f];
                new_row_indices[index] = i;
                tmp_offset[f]++;
            }
        }

        // free(tmp_offset);

        std::swap(this->n_points, this->n_filters);

        // delete[] this->row_offsets;
        // delete[] this->row_indices;

        this->row_offsets = std::move(new_row_offsets);
        this->row_indices = std::move(new_row_indices);

        this->transposed = ~transposed;
        return;
    }

    /* if the filters object is transposed, returns an untransposed version */
    csr_filters reverse_transpose() {
        if (~transposed) {
            std::cout << "This csr_filters is not transposed" << std::endl;
            return *this;
        }
        transposed = false;
        csr_filters out = transpose();
        out.transposed = false;
        return std::move(out);
    }

    // csr_filters copy() {
    //     auto out = csr_filters(n_points, n_filters, n_nonzero, row_offsets, row_indices);
    //     out.transposed = transposed;
    //     return out;
    // }

    /* subsets the rows based on the indices

        if you then transpose the output you can get the inverted index for a posting list

        Similarly, if you want to only track specific filters, you can subset after transposing

        keep in  mind:
            - the indices of the rows have to be mapped with the indices sequence to get back to the original values (but the filter numbers remain the same)
            - This is a copy, not a view
     */
    csr_filters subset_rows(parlay::sequence<int32_t> indices) const {
        // int64_t* new_row_offsets = (int64_t*) malloc((indices.size() + 1) * sizeof(int64_t)); // where to index for each filter (length is +1 because the last value is nnz to make the length calculation work for the last one)
        std::unique_ptr<int64_t[]> new_row_offsets = std::make_unique<int64_t[]>(indices.size() + 1);
        new_row_offsets[0] = 0;
        for (int64_t i = 0; i < indices.size(); i++) {
            new_row_offsets[i + 1] = row_offsets[indices[i] + 1] - row_offsets[indices[i]] + new_row_offsets[i];
        }

        // int32_t* new_row_indices = (int32_t*) malloc(new_row_offsets[indices.size()] * sizeof(int32_t)); // indices of matching points
        std::unique_ptr<int32_t[]> new_row_indices = std::make_unique<int32_t[]>(new_row_offsets[indices.size()]);

        for (int64_t i = 0; i < indices.size(); i++) {
            memcpy(new_row_indices.get() + new_row_offsets[i], row_indices.get() + row_offsets[indices[i]], (row_offsets[indices[i] + 1] - row_offsets[indices[i]]) * sizeof(int32_t));
        }

        auto out = csr_filters();
        out.n_points = indices.size();
        out.n_filters = n_filters;
        out.n_nonzero = new_row_offsets[indices.size()];
        out.row_offsets = std::move(new_row_offsets);
        out.row_indices = std::move(new_row_indices);
        out.transposed = transposed;
        return out;
    }

    /* Copies to a new csr_filters with only the specified filters (columns)

    As written preserves the number of the filters (or points if transposed), which should be much easier to work with
     */
    csr_filters subset_filters(parlay::sequence<int32_t> filters) const {
        // construct a boolean array of which filters to keep
        // bool* keep = (bool*) malloc(n_filters * sizeof(bool));
        std::unique_ptr<bool[]> keep = std::make_unique<bool[]>(n_filters);
        memset(keep.get(), false, n_filters * sizeof(bool));
        for (int64_t i = 0; i < filters.size(); i++) {
            keep[filters[i]] = true;
        }

        // compute the new offsets
        // int64_t* new_row_offsets = (int64_t*) malloc((n_points + 1) * sizeof(int64_t)); // where to index for each filter (length is +1 because the last value is nnz to make the length calculation work for the last one)
        std::unique_ptr<int64_t[]> new_row_offsets = std::make_unique<int64_t[]>(n_points + 1);
        memset(new_row_offsets.get(), 0, (n_points + 1) * sizeof(int64_t)); // initializing to 0s
        for (int64_t i = 0; i < n_points; i++) {
            int64_t start = row_offsets[i];
            int64_t end = row_offsets[i + 1];
            new_row_offsets[i + 1] = new_row_offsets[i];
            for (int64_t j = start; j < end; j++) {
                if (keep[row_indices[j]]) {
                    new_row_offsets[i + 1]++;
                }
            }
        }

        // compute the new indices
        // int32_t* new_row_indices = (int32_t*) malloc(new_row_offsets[n_points] * sizeof(int32_t)); // indices of matching points
        std::unique_ptr<int32_t[]> new_row_indices = std::make_unique<int32_t[]>(new_row_offsets[n_points]);
        // this could just iterate over the indices but this is more readable
        for (int64_t i = 0; i < n_points; i++) {
            int64_t start = row_offsets[i];
            int64_t end = row_offsets[i + 1];
            int64_t new_index = new_row_offsets[i];
            for (int64_t j = start; j < end; j++) {
                if (keep[row_indices[j]]) {
                    new_row_indices[new_index] = row_indices[j];
                    new_index++;
                }
            }
        }

        // delete keep;

        auto out = csr_filters();
        out.n_points = n_points;
        out.n_filters = filters.size();
        out.n_nonzero = new_row_offsets[n_points];
        out.row_offsets = std::move(new_row_offsets);
        out.row_indices = std::move(new_row_indices);
        out.transposed = transposed;
        return out;
    }

    /*
    This abominably named function does 3 things:
        - subsets the rows based on the indices
        - transposes the result
        - renames the columns to match the indices

    Internally, the implementation does not do all 3 of these things independently, but the result should be the same.

    sorts the indices just in case they aren't already sorted
     */
    csr_filters subset_rows_transpose(parlay::sequence<int32_t>& indices) const {
        std::sort(indices.begin(), indices.end()); // strictly speaking, this shouldn't violate constness of the reference

        int64_t new_n_points = n_filters;
        int64_t new_n_filters = indices.size();
        int64_t new_n_nonzero = 0;

        for (int64_t i = 0; i < indices.size(); i++) {
            new_n_nonzero += row_offsets[indices[i] + 1] - row_offsets[indices[i]];
        }

        std::unique_ptr<int64_t[]> new_row_offsets = std::make_unique<int64_t[]>(new_n_points + 1);
        std::unique_ptr<int32_t[]> new_row_indices = std::make_unique<int32_t[]>(new_n_nonzero);

        // we use new_row_offsets to count the number of times each filter appears in the new data, then we can scan it to get the offsets
        memset(new_row_offsets.get(), 0, (new_n_points + 1) * sizeof(int64_t)); // initializing to 0s

        for (int64_t i = 0; i < indices.size(); i++) {
            int64_t start = row_offsets[indices[i]];
            int64_t end = row_offsets[indices[i] + 1];
            for (int64_t j = start; j < end; j++) {
                new_row_offsets[row_indices[j] + 1]++;
            }
        }

        // not a sequence so doing this serially
        for (int64_t i = 1; i < new_n_points + 1; i++) {
            new_row_offsets[i] += new_row_offsets[i - 1];
        }

        std::unique_ptr<int64_t[]> tmp_offset = std::make_unique<int64_t[]>(new_n_points);
        memset(tmp_offset.get(), 0, new_n_points * sizeof(int64_t)); // initializing to 0s

        for (int64_t i = 0; i < indices.size(); i++) {
            int64_t start = row_offsets[indices[i]];
            int64_t end = row_offsets[indices[i] + 1];
            for (int64_t j = start; j < end; j++) {
                int64_t f = row_indices[j];
                int64_t index = new_row_offsets[f] + tmp_offset[f];
                new_row_indices[index] = i;
                tmp_offset[f]++;
            }
        }

        auto out = csr_filters();
        out.n_points = new_n_points;
        out.n_filters = new_n_filters;
        out.n_nonzero = new_n_nonzero;
        out.row_offsets = std::move(new_row_offsets);
        out.row_indices = std::move(new_row_indices);
        out.transposed = true;
        return out;
    }

    parlay::sequence<int32_t> nonempty_rows() const {
        return parlay::filter(parlay::iota<int32_t>(n_points), [&] (int32_t i) {
            return row_offsets[i + 1] - row_offsets[i] > 0;
        });
    }
};

// /* Creating this struct just for the sake of having a complete implementation if we ever need to use csr where we care about the values */
// struct csr_matrix{
//     int64_t num_points;
//     int64_t num_filters;
//     int64_t num_nonzeros;
//     int64_t* row_offsets; // indices into data
//     int32_t* row_indices; // the indices of the nonzero entries
//     float* values; // the values of the nonzero entries

//     /* mmaps filter data in csr form from filename */
//     csr_filters(char* filename) {
//         // opening file stream
//         FILE* fp = fopen(filename, "rb");
//         if (fp == NULL) {
//             fprintf(stderr, "Error opening file %s\n", filename);
//             exit(1);
//         }

//         // reading in number of points, filters, and nonzeros
//         fread(&num_points, sizeof(int64_t), 1, fp);
//         fread(&num_filters, sizeof(int64_t), 1, fp);
//         fread(&num_nonzeros, sizeof(int64_t), 1, fp);

//         // reading in row offsets
//         row_offsets = (int64_t*) malloc((num_points + 1) * sizeof(int64_t));
//         fread(row_offsets, sizeof(int64_t), num_points + 1, fp);

//         // reading in row lengths
//         row_indices = (int32_t*) malloc(num_nonzeros * sizeof(int32_t));
//         fread(row_indices, sizeof(int32_t), num_nonzeros, fp);

//         // reading in values
//         values = (float*) malloc(num_nonzeros * sizeof(float));
//         fread(values, sizeof(float), num_nonzeros, fp);

//         fclose(fp);
//     }
// }


#endif // CSR_FILTER_H