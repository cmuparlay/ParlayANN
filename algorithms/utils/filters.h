/* interface for point filters represented in CSR format */
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


struct csr_filters{
    int64_t n_points;
    int64_t n_filters;
    int64_t n_nonzero;
    int64_t* row_offsets; // indices into data
    int32_t* row_indices; // the indices of the nonzero entries, which is actually all we need
    
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
        row_offsets = (int64_t*) malloc((n_points + 1) * sizeof(int64_t));
        fread(row_offsets, sizeof(int64_t), n_points + 1, fp);

        // reading in row lengths
        row_indices = (int32_t*) malloc(n_nonzero * sizeof(int32_t));
        fread(row_indices, sizeof(int32_t), n_nonzero, fp);

        fclose(fp);
    }

    /* constructs csr_filters with all the values already provided as arguments (should probably be protected)*/
    csr_filters(int64_t n_points, int64_t n_filters, int64_t n_nonzero, int64_t* row_offsets, int32_t* row_indices) : n_points(n_points), n_filters(n_filters), n_nonzero(n_nonzero), row_offsets(row_offsets), row_indices(row_indices) {}

    void del() {
        free(row_offsets);
        free(row_indices);
    }

    void print_stats(){
        printf("n_points: %ld\n", n_points);
        printf("n_filters: %ld\n", n_filters);
        printf("n_nonzeros: %ld\n", n_nonzero);
    }

    /* Returns true if p matches filter f, which is equivalent to row p column i being nonzero */
    bool match(int64_t p, int64_t f) {
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

    /* I would like to be able to get the filters associated with a point in a python-accesible way but this is a good enough proof of concept until I figure out how to do that */
    int64_t first_label(int64_t p) {
        return row_indices[row_offsets[0]];
    }

    /* Returns the number of points matching a given filter */
    int64_t filter_count(int64_t f) {
        return parlay::reduce(parlay::delayed_seq<int64_t>(n_points, [&] (int64_t i) {
            return match(i, f);
        }));
    }

    /* Returns the number of filters associated with a point */
    int64_t point_count(int64_t p) {
        return row_offsets[p + 1] - row_offsets[p];
    }

    parlay::sequence<int64_t> filter_counts() {
        return parlay::tabulate(n_filters, [&] (int64_t i) {
            return filter_count(i);
        });
    }

    /* Transposes to make acessing points associated with a filter fast */
    csr_filters transpose() {
        int64_t* new_row_offsets = (int64_t*) malloc((n_filters + 1) * sizeof(int64_t)); // where to index for each filter (length is +1 because the last value is nnz to make the length calculation work for the last one)
        int32_t* new_row_indices = (int32_t*) malloc(n_nonzero * sizeof(int32_t)); // indices of matching points

        // initializing both arrays to 0s
        memset(new_row_offsets, 0, (n_filters + 1) * sizeof(int64_t));
        memset(new_row_indices, 0, n_nonzero * sizeof(int32_t));
        
        // should only need to iterate once
        for (int64_t i = 0; i < n_points; i++) {
            int64_t start = row_offsets[i];
            int64_t end = row_offsets[i + 1];
            for (int64_t j = start; j < end; j++) {
                int64_t f = row_indices[j];
                int64_t index = new_row_offsets[f];
                new_row_indices[index] = i;
                new_row_offsets[f]++;
            }
        }

        return csr_filters(n_filters, n_points, n_nonzero, new_row_offsets, new_row_indices);
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