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