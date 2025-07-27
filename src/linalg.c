#include "linalg.h"

#ifdef _MSC_VER
    #include <intrin.h>
#else
    #include <cpuid.h>
#endif

#include <immintrin.h>
#include <stdint.h>
#include <math.h>

LinAlgFuncs linalg_funcs;

void init_linalg() {
    if (has_avx512_support()) {
        linalg_funcs.matrix_vector_mult = matrix_vector_mult_avx512;
        linalg_funcs.matrix_transpose_vector_mult = matrix_transpose_vector_mult_avx512;
        linalg_funcs.outer_product_add = outer_product_add_avx512;
    } else {
        linalg_funcs.matrix_vector_mult = matrix_vector_mult_scalar;
        linalg_funcs.matrix_transpose_vector_mult = matrix_transpose_vector_mult_scalar;
        linalg_funcs.outer_product_add = outer_product_add_scalar;   
    }
}

void matrix_vector_mult_avx512(const float* matrix, const float* vector, float* result, const int m, const int n) {
    for (int i = 0; i < m; i++) {
        int j;
        __m512 v_sum = _mm512_setzero_ps();
        
        for (j = 0; j <= n - 16; j += 16) {
            const __m512 v_matrix = _mm512_load_ps(&matrix[i * n + j]);
            const __m512 v_vector = _mm512_load_ps(&vector[j]);
    
            v_sum = _mm512_fmadd_ps(v_matrix, v_vector, v_sum);
        }
        
        result[i] = _mm512_reduce_add_ps(v_sum);

        for (; j < n; j++) {
            result[i] += matrix[i * n + j] * vector[j];
        }
    }
}

void matrix_vector_mult_scalar(const float *matrix, const float *vector, float *result, int m, int n) {
    for (int i = 0; i < m; i++) {
        result[i] = 0.0f;
    
        for (int j = 0; j < n; j++) {
            result[i] += matrix[i * n + j] * vector[j];
        }   
    }
}

void matrix_vector_mult(const float *matrix, const float *vector, float *result, const int m, const int n) {
    linalg_funcs.matrix_vector_mult(matrix, vector, result, m, n);
}

void matrix_transpose_vector_mult_avx512(const float* matrix, const float* vector, float* result, const int m, const int n) {
    for (int i = 0; i < n; i++) {
        result[i] = 0.0f;
    }

    for (int i = 0; i < m; i++) {
        int j;
        const __m512 v_vector = _mm512_set1_ps(vector[i]);
        
        for (j = 0; j <= n - 16; j += 16) {
            const __m512 v_matrix = _mm512_load_ps(&matrix[i * n + j]);
            __m512 v_result = _mm512_load_ps(&result[j]);
    
            v_result = _mm512_fmadd_ps(v_matrix, v_vector, v_result);
    
            _mm512_store_ps(&result[j], v_result);
        }
    
        for (; j < n; j++) {
            result[j] += matrix[i * n + j] * vector[i];   
        }
    }
}

void matrix_transpose_vector_mult_scalar(const float* matrix, const float* vector, float* result, const int m, const int n) {
    for (int i = 0; i < n; i++) {
        result[i] = 0.0f;
    }

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[j] += matrix[i * n + j] * vector[i];
        }
    }
}

void matrix_transpose_vector_mult(const float *matrix, const float *vector, float *result, const int m, const int n) {
    linalg_funcs.matrix_transpose_vector_mult(matrix, vector, result, m, n);
}

void outer_product_add_avx512(float* matrix, const float* vec_m, const float* vec_n, const float scalar, const int m, const int n) {
    for (int i = 0; i < m; i++) {
        int j;
        const __m512 v_vec_m = _mm512_set1_ps(vec_m[i] * scalar);
        
        for (j = 0; j <= n - 16; j += 16) {
            __m512 v_matrix = _mm512_load_ps(&matrix[i * n + j]);
            const __m512 v_vec_n = _mm512_load_ps(&vec_n[j]);
    
            v_matrix = _mm512_fmadd_ps(v_vec_m, v_vec_n, v_matrix);
    
            _mm512_store_ps(&matrix[i * n + j], v_matrix);
        }
    
        for (; j < n; j++) {
            matrix[i * n + j] += vec_m[i] * vec_n[j] * scalar;
        }
    }
}

void outer_product_add_scalar(float* matrix, const float* vec_m, const float* vec_n, const float scalar, const int m, const int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] += vec_m[i] * vec_n[j] * scalar;
        }
    }
}

void outer_product_add(float *matrix, const float *vec_m, const float *vec_n, const float scalar, const int m, const int n) {
    linalg_funcs.outer_product_add(matrix, vec_m, vec_n, scalar, m, n);
}

void vector_add(const float* a, const float* b, float* out, const int n) {
    // int i;
    // for (i = 0; i <= n - 16; i += 16) {
    //     __m512 v_a = _mm512_load_ps(&a[i]);
    //     __m512 v_b = _mm512_load_ps(&b[i]);
    //
    //     v_a = _mm512_add_ps(v_a, v_b);
    //     
    //     _mm512_store_ps(&out[i], v_a);
    // }
    //
    // for (; i < n; i++) {
    //     out[i] = a[i] + b[i];
    // }
    
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void vector_add_scaled(const float *a, const float *b, float *out, const float scalar_b, const int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i] * scalar_b;
    }
}

void vector_sub(const float *a, const float *b, float *out, const int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] - b[i];
    }
}

void hadamard_product(const float* a, const float* b, float* out, const int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] * b[i];
    }
}

void relu(const float* input, float* output, const int n) {
    for (int i = 0; i < n; i++) {
        if (input[i] < 0) {
            output[i] = 0;
        } else {
            output[i] = input[i];
        }
    }
}

void relu_d(const float* input, float* output, const int n) {
    for (int i = 0; i < n; i++) {
        if (input[i] <= 0) {
            output[i] = 0;
        } else {
            output[i] = 1;
        }
    }
}

void sigmoid(const float* input, float* output, const int n) {
    for (int i = 0; i < n; i++) {
        output[i] = 1.0f / (1.0f + exp(-input[i]));
    }
}

void sigmoid_d(const float* input, float* output, const int n) {
    for (int i = 0; i < n; i++) {
        output[i] = input[i] * (1.0f - input[i]);
    }  
}

int has_avx512_support() {
    // Check if CPU supports AVX-512
    unsigned int eax, ebx, ecx, edx;
#ifdef _MSC_VER
    int info[4];
    __cpuid(info, 0);
    if (info[0] < 7)
        return 0;

    __cpuidex(info, 7, 0);
    ebx = info[1];
#else
    if (__get_cpuid_max(0, NULL) < 7)
        return 0;

    __cpuid_count(7, 0, eax, ebx, ecx, edx);
#endif

    if ((ebx & (1 << 16)) == 0)
        return 0;

    // Check if OS supports AVX-512
    uint64_t xcr_feature_mask;
#ifdef _MSC_VER
    xcr_feature_mask = _xgetbv(0);
#else
    uint32_t xcr_eax, xcr_edx;
    __asm__ volatile(".byte 0x0f, 0x01, 0xd0" : "=a"(xcr_eax), "=d"(xcr_edx) : "c"(0));
    xcr_feature_mask = ((uint64_t) xcr_edx << 32) | xcr_eax;
#endif

    const uint64_t required_bits = 1 << 1 | 1 << 2 | 1 << 5 | 1 << 6 | 1 << 7;
    return (xcr_feature_mask & required_bits) == required_bits;   
}
