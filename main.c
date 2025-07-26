#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_OUTPUTS 10
#define LEARNING_RATE 0.001f

int flip_endianness(const uint8_t* input) {
    return input[0] << 24 | input[1] << 16 | input[2] << 8 | input[3];
}

uint8_t* read_labels(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file!\n");
        return NULL;
    }

    uint8_t header[8];
    fread(header, sizeof(int), 2, file);

    int magic_number = flip_endianness(header);
    int n = flip_endianness(header + 4);
    // printf("test labels magic number: %d\n", magic_number);
    if (magic_number != 2049) {
        fprintf(stderr, "Invalid magic number!\n");
        return NULL;
    }
    
    // printf("number of test labels: %d\n", n);

    uint8_t* labels = malloc(n * sizeof(uint8_t));
    fread(labels, sizeof(uint8_t), n, file);

    return labels;
}

typedef struct {
    uint8_t* data;
    int rows;
    int cols;
    int length;
} Images;

Images* read_tests(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file!\n");
        return NULL;
    }

    uint8_t header[16];
    fread(header, sizeof(int), 4, file);

    const int magic_number = flip_endianness(header);
    const int length = flip_endianness(header + 4);
    const int rows = flip_endianness(header + 8);
    const int cols = flip_endianness(header + 12);

    // printf("test images magic number: %d\n", magic_number);
    if (magic_number != 2051) {
        fprintf(stderr, "Invalid magic number!\n");
        return NULL;
    }
    
    // printf("number of test images: %d\n", length);
    // printf("number of rows: %d\n", rows);
    // printf("number of cols: %d\n", cols);

    uint8_t* data = malloc(length * rows * cols * sizeof(uint8_t));
    fread(data, sizeof(uint8_t), length * rows * cols, file);

    Images* images = malloc(sizeof(Images));
    images->data = data;
    images->rows = rows;
    images->cols = cols;
    images->length = length;

    return images;
}

typedef struct {
    int size;
    int prev_size;
    float* weights;
    float* biases;
    float* activation;
} NNLayer;

void free_layers(NNLayer* layers, const int num_layers) {
    for (int i = 0; i < num_layers; i++) {
        free(layers[i].weights);
        free(layers[i].biases);
        free(layers[i].activation);
    }
    
    free(layers);   
}

float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

float he_uniform_init(const int fan_in) {
    const float limit = sqrtf(6.0f / (float) fan_in);
    return rand_float() * 2.0f * limit - limit;
}

NNLayer* create_layers(const int* layer_sizes, const int num_layers) {
    NNLayer* layers = malloc(sizeof(NNLayer) * num_layers);
    
    for (int i = 0; i < num_layers; i++) {
        layers[i].size = layer_sizes[i + 1];
        layers[i].prev_size = layer_sizes[i];
        layers[i].weights = malloc(sizeof(float) * layer_sizes[i] * layer_sizes[i + 1]);
        layers[i].biases = malloc(sizeof(float) * layer_sizes[i + 1]);
        layers[i].activation = malloc(sizeof(float) * layer_sizes[i + 1]);

        for (int j = 0; j < layers[i].size * layers[i].prev_size; j++) {
            layers[i].weights[j] = he_uniform_init(layers[i].prev_size);
        }

        for (int j = 0; j < layers[i].size; j++) {
            layers[i].biases[j] = 0.0f;
        }
    }

    return layers;
}

void matrix_vector_mult(const float* matrix, const float* vector, float* result, const int m, const int n) {
    for (int i = 0; i < m; i++) {
        result[i] = 0.0f;
        
        for (int j = 0; j < n; j++) {
            result[i] += matrix[i * n + j] * vector[j];
        }
    }
}

void matrix_transpose_vector_mult(const float* matrix, const float* vector, float* result, const int m, const int n) {
    for (int i = 0; i < n; i++) {
        result[i] = 0.0f;
    }
    
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            result[j] += matrix[i * n + j] * vector[i];
        }
    }
}

void outer_product_add(float* matrix, const float* vec_m, const float* vec_n, const float scalar, const int m, const int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i * n + j] += vec_m[i] * vec_n[j] * scalar;
        }
    }  
}

void vector_add(const float* a, const float* b, float* out, const int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void vector_add_in_place_scaled(float* dest, const float* src, const float scalar, const int n) {
    for (int i = 0; i < n; i++) {
        dest[i] += src[i] * scalar;
    }
}

void vector_sub(const float* a, const float* b, float* out, const int n) {
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

void forwardprop(const NNLayer* layers, const int num_layers, const float* input) {
    for (int i = 0; i < num_layers; i++) {
        // Compute activation
        if (i == 0) {
            matrix_vector_mult(layers[i].weights, input, layers[i].activation, layers[i].size, layers[i].prev_size);
        } else {
            matrix_vector_mult(layers[i].weights, layers[i - 1].activation, layers[i].activation, layers[i].size, layers[i].prev_size);
        }
        
        // Add bias
        vector_add(layers[i].activation, layers[i].biases, layers[i].activation, layers[i].size);
        
        // Apply ReLU
        if (i < num_layers - 1) {
            relu(layers[i].activation, layers[i].activation, layers[i].size);
        } else {
            sigmoid(layers[i].activation, layers[i].activation, layers[i].size);
        }
    }
}

void generate_expected_vec(float* expected_vec, const int expected) {

    for (int i = 0; i < NUM_OUTPUTS; i++) {
        expected_vec[i] = i == expected ? 1.0f : 0.0f;
    }
}  

float cost(const float* nn_output, const float* expected) {
    float ssd = 0.0f;

    for (int i = 0; i < NUM_OUTPUTS; i++) {
        const float diff = nn_output[i] - expected[i];
        ssd += diff * diff;
    }

    return ssd;
}

void backprop(NNLayer* layers, const int num_layers, const int max_layer_size, const float* input, const float* expected) {
    NNLayer* output_layer = &layers[num_layers - 1];
    float* layer_errors[num_layers];
    for (int i = 0; i < num_layers; i++) {
        layer_errors[i] = malloc(sizeof(float) * layers[i].size);   
    }

    float activation_func_error[max_layer_size];

    // Compute δ^L = (a^L - y) * (σ'(z^L)) for the output layer
    vector_sub(output_layer->activation, expected, layer_errors[num_layers - 1], output_layer->size);
    sigmoid_d(output_layer->activation, activation_func_error, output_layer->size);
    hadamard_product(layer_errors[num_layers - 1], activation_func_error, layer_errors[num_layers - 1], output_layer->size);

    // Wait to update weights for layer l until after the next layers error has been computed.
    // The next layers error is the only thing that depends on the current weights.
    for (int i = num_layers - 2; i >= 0; i--) {
        // Compute δ^l = (w^{l+1} * δ^{l+1}) * (σ'(z^l)) for the hidden layer
        matrix_transpose_vector_mult(layers[i + 1].weights, layer_errors[i + 1], layer_errors[i], layers[i + 1].size, layers[i].size);
        relu_d(layers[i].activation, activation_func_error, layers[i].size);
        hadamard_product(layer_errors[i], activation_func_error, layer_errors[i], layers[i].size);
        
        // Update weights for layer l + 1
        // outer_product_add(layers[i + 1].weights, layers[i].activation, layer_errors[i + 1], -LEARNING_RATE, layers[i].size, layers[i + 1].size);
        outer_product_add(layers[i + 1].weights, layer_errors[i + 1], layers[i].activation, -LEARNING_RATE, layers[i + 1].size, layers[i].size);

        // Update biases for layer l + 1
        vector_add_in_place_scaled(layers[i + 1].biases, layer_errors[i + 1], -LEARNING_RATE, layers[i + 1].size);
    }

    // Compute weights and biases for the first hidden layer
    // outer_product_add(layers[0].weights, input, layer_errors[0], -LEARNING_RATE, layers[0].prev_size, layers[0].size);
    outer_product_add(layers[0].weights, layer_errors[0], input, -LEARNING_RATE, layers[0].size, layers[0].prev_size);
    vector_add_in_place_scaled(layers[0].biases, layer_errors[0], -LEARNING_RATE, layers[0].size);

    for (int i = 0; i < num_layers; i++) {
        free(layer_errors[i]);
    }
}

int main() {
    srand(time(NULL));
    
    uint8_t* labels = read_labels("../train-labels.idx1-ubyte");
    Images* images = read_tests("../train-images.idx3-ubyte");

    if (labels == NULL || images == NULL) {
        exit(EXIT_FAILURE);
    }

    printf("Read all tests!\n");

    const int layer_sizes[] = {images->rows * images->cols, 128, 64, NUM_OUTPUTS};
    const int num_layers = 3; // TODO: Move these to defines and stop hardcoding these values
    const int max_layer_size = 128;
    NNLayer* layers = create_layers(layer_sizes, 3);

    float nn_input[images->rows * images->cols];
    float expected[NUM_OUTPUTS];

    printf("Length: %d\n", images->length);

    for (int epoch = 0; epoch < 10; epoch ++) {
        for (int i = 0; i < images->length; i++) {
            generate_expected_vec(expected, labels[i]);

            for (int j = 0; j < layers[0].prev_size; j++) {
                nn_input[j] = (float) images->data[i * layers[0].prev_size + j] / (float) 255;
            }

            forwardprop(layers, num_layers, nn_input);
            backprop(layers, num_layers, max_layer_size, nn_input, expected);
        }

        printf("Epoch %d finished\n", epoch + 1);
    }

    const uint8_t* test_labels = read_labels("../t10k-labels.idx1-ubyte");
    const Images* test_images = read_tests("../t10k-images.idx3-ubyte");

    int total = 0;
    int correct = 0;
    for (int i = 0; i < test_images->length; i++) {
        for (int j = 0; j < layers[0].prev_size; j++) {
            nn_input[j] = (float) test_images->data[i * layers[0].prev_size + j] / (float) 255;
        }

        forwardprop(layers, num_layers, nn_input);
        const float* out = layers[num_layers - 1].activation;

        float max_out = 0.0f;
        int max_out_index = -1;
        for (int j = 0; j < NUM_OUTPUTS; j++) {
            if (out[j] > max_out) {
                max_out = out[j];
                max_out_index = j;
            }
        }

        total++;
        if (max_out_index == test_labels[i]) {
            correct++;
        }
    }

    printf("Total: %d\n", total);
    printf("Correct: %d\n", correct);
    printf("Accuracy: %f\n", (float) correct / (float) total);
    
    free_layers(layers, num_layers);
    free(labels);
    free(images->data);
    free(images);
    
    return 0;
}
