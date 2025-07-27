#ifndef LINALG_H
#define LINALG_H

typedef struct {
    void (*matrix_vector_mult)(const float*, const float*, float*, int, int);
    void (*matrix_transpose_vector_mult)(const float*, const float*, float*, int, int);
    void (*outer_product_add)(float*, const float*, const float*, float, int, int);
} LinAlgFuncs;

extern LinAlgFuncs linalg_funcs;

void init_linalg();

static void matrix_vector_mult_avx512(const float* matrix, const float* vector, float* result, int m, int n);
static void matrix_vector_mult_scalar(const float* matrix, const float* vector, float* result, int m, int n);
void matrix_vector_mult(const float* matrix, const float* vector, float* result, int m, int n);

static void matrix_transpose_vector_mult_avx512(const float* matrix, const float* vector, float* result, int m, int n);
static void matrix_transpose_vector_mult_scalar(const float* matrix, const float* vector, float* result, int m, int n);
void matrix_transpose_vector_mult(const float* matrix, const float* vector, float* result, int m, int n);

static void outer_product_add_avx512(float* matrix, const float* vec_m, const float* vec_n, float scalar, int m, int n);
static void outer_product_add_scalar(float* matrix, const float* vec_m, const float* vec_n, float scalar, int m, int n);
void outer_product_add(float* matrix, const float* vec_m, const float* vec_n, float scalar, int m, int n);

void vector_add(const float* a, const float* b, float* out, int n);
void vector_sub(const float* a, const float* b, float* out, int n);
void vector_add_scaled(const float* a, const float* b, float* out, float scalar_b, int n);
void hadamard_product(const float* a, const float* b, float* out, int n);

void relu(const float* input, float* output, int n);
void relu_d(const float* input, float* output, int n);
void sigmoid(const float* input, float* output, int n);
void sigmoid_d(const float* input, float* output, int n);

int has_avx512_support();

#endif //LINALG_H
