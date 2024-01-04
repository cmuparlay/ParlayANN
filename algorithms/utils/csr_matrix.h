/* A general-purpose binary csr matrix implementation 

will probably write a real csr matrix implementation later*/
#ifndef CSR_MATRIX_H
#define CSR_MATRIX_H

#include "parlay/sequence.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility>
#include <stdint.h>

template <typename row_type, typename col_type>
struct BinaryCSRMatrix {
    row_type n_rows;
    col_type n_cols;
    size_t n_nonzero;
    std::unique_ptr<row_type[]> row_offsets;
    std::unique_ptr<col_type[]> row_indices;

    BinaryCSRMatrix() {}

    /* currently assumes the leading values are provided as int64_t, and if you want this to work with existing files, row_type should be int64_t and col_type should be int32_t*/
    BinaryCSRMatrix(std::string filename) {
        FILE *f = fopen(filename.c_str(), "r");
        if(f == NULL) {
            fprintf(stderr, "Error: could not open file %s\n", filename.c_str());
            exit(1);
        }

        int64_t n_rows, n_cols, n_nonzero;
        fread(&n_rows, sizeof(int64_t), 1, f);
        fread(&n_cols, sizeof(int64_t), 1, f);
        fread(&n_nonzero, sizeof(int64_t), 1, f);

        this->n_rows = n_rows;
        this->n_cols = n_cols;
        this->n_nonzero = n_nonzero;
        
        // reading in row offsets
        row_offsets = std::unique_ptr<row_type[]>(new row_type[n_rows+1]);
        fread(row_offsets.get(), sizeof(row_type), n_rows+1, f);

        // reading in column indices
        row_indices = std::unique_ptr<col_type[]>(new col_type[n_nonzero]);
        fread(row_indices.get(), sizeof(col_type), n_nonzero, f);

        fclose(f);
    }

    // BinaryCSRMatrix<col_type, row_type> transpose() {
    //     std::unique_ptr<row_type[]> new_row_offsets = std::unique_ptr<row_type[]>(new row_type[n_cols+1]);
    //     std::unique_ptr<col_type[]> new_row_indices = std::unique_ptr<col_type[]>(new col_type[n_nonzero]);
    // }
};

#endif // CSR_MATRIX_H