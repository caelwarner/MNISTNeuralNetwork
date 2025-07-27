#include "main.h"

#include <math.h>
#include <time.h>
#include <immintrin.h>

#include "io.h"
#include "linalg.h"

float he_uniform_init(const int fan_in) {
    const float limit = sqrtf(6.0f / (float) fan_in);
    return rand_float() * 2.0f * limit - limit;
}

NNLayer* create_layers(const int input_layer_size, const int* layer_sizes, const int num_layers) {
    NNLayer* layers = malloc(sizeof(NNLayer) * num_layers);
    
    for (int i = 0; i < num_layers; i++) {
        layers[i].size = layer_sizes[i];
        if (i == 0) {
            layers[i].prev_size = input_layer_size;
        } else {
            layers[i].prev_size = layer_sizes[i - 1];
        }
        
        layers[i].weights = _mm_malloc(sizeof(float) * layers[i].size * layers[i].prev_size, 64);
        layers[i].biases = _mm_malloc(sizeof(float) * layers[i].size, 64);
        layers[i].activation = _mm_malloc(sizeof(float) * layers[i].size, 64);

        for (int j = 0; j < layers[i].size * layers[i].prev_size; j++) {
            layers[i].weights[j] = he_uniform_init(layers[i].prev_size);
        }

        for (int j = 0; j < layers[i].size; j++) {
            layers[i].biases[j] = 0.0f;
        }
    }

    return layers;
}

void free_layers(NNLayer* layers, const int num_layers) {
    for (int i = 0; i < num_layers; i++) {
        _mm_free(layers[i].weights);
        _mm_free(layers[i].biases);
        _mm_free(layers[i].activation);
    }
    
    free(layers);   
}

float* get_layer_output(const NNLayer* layers, const int num_layers) {
    return layers[num_layers - 1].activation;
}

void generate_expected_vec(float* expected_vec, const int expected) {
    for (int i = 0; i < NUM_OUTPUTS; i++) {
        expected_vec[i] = i == expected ? 1.0f : 0.0f;
    }
}

void forward_propagation(NNLayer* layers, const int num_layers, const float* input) {
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

void backward_propagation(NNLayer* layers, const int num_layers, const int max_layer_size, const float* input, const float* expected) {
    const NNLayer* output_layer = &layers[num_layers - 1];
    // float* layer_errors[num_layers];
    float** layer_errors = malloc(sizeof(float*) * num_layers);
    for (int i = 0; i < num_layers; i++) {
        layer_errors[i] = _mm_malloc(sizeof(float) * layers[i].size, 64);   
    }

    float* activation_func_error = malloc(sizeof(float) * max_layer_size);

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
        vector_add_scaled(layers[i + 1].biases, layer_errors[i + 1], layers[i + 1].biases, -LEARNING_RATE, layers[i + 1].size);
    }

    // Compute weights and biases for the first hidden layer
    // outer_product_add(layers[0].weights, input, layer_errors[0], -LEARNING_RATE, layers[0].prev_size, layers[0].size);
    outer_product_add(layers[0].weights, layer_errors[0], input, -LEARNING_RATE, layers[0].size, layers[0].prev_size);
    vector_add_scaled(layers[0].biases, layer_errors[0], layers[0].biases, -LEARNING_RATE, layers[0].size);

    for (int i = 0; i < num_layers; i++) {
        _mm_free(layer_errors[i]);
    }
    free(layer_errors);
    free(activation_func_error);
}

float cost(const float* nn_output, const float* expected) {
    float ssd = 0.0f;

    for (int i = 0; i < NUM_OUTPUTS; i++) {
        const float diff = nn_output[i] - expected[i];
        ssd += diff * diff;
    }

    return ssd;
}

int main() {
    init_linalg();
    srand(time(NULL));
    
    uint8_t* labels = read_labels("../train-labels.idx1-ubyte");
    Images* images = read_images("../train-images.idx3-ubyte");

    if (labels == NULL || images == NULL) {
        exit(EXIT_FAILURE);
    }

    printf("Read all tests!\n");

    int max_layer_size = 0;
    for (int i = 0; i < NUM_LAYERS; i++) {
        if (LAYER_SIZES[i] > max_layer_size) {
            max_layer_size = LAYER_SIZES[i];
        }
    }

    if (max_layer_size == 0) {
        fprintf(stderr, "Error: There are no layers with a size greater than 0\n");
        exit(EXIT_FAILURE);
    }
    
    NNLayer* layers = create_layers(images->rows * images->cols, LAYER_SIZES, NUM_LAYERS);

    // float nn_input[images->rows * images->cols];
    float* nn_input = _mm_malloc(sizeof(float) * images->rows * images->cols, 64);
    float expected[NUM_OUTPUTS];

    printf("Length: %d\n", images->length);

    clock_t start = clock();
    
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch ++) {
        for (int i = 0; i < images->length; i++) {
            generate_expected_vec(expected, labels[i]);

            for (int j = 0; j < layers[0].prev_size; j++) {
                nn_input[j] = (float) images->data[i * layers[0].prev_size + j] / (float) 255;
            }

            forward_propagation(layers, NUM_LAYERS, nn_input);
            backward_propagation(layers, NUM_LAYERS, max_layer_size, nn_input, expected);
        }

        printf("Epoch %d finished\n", epoch + 1);
    }

    clock_t end = clock();
    double cpu_time = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Training took: %f\n", cpu_time);
    
    uint8_t* test_labels = read_labels("../t10k-labels.idx1-ubyte");
    Images* test_images = read_images("../t10k-images.idx3-ubyte");

    if (test_labels == NULL || test_images == NULL) {
        exit(EXIT_FAILURE);
    }
    
    start = clock();
    
    int total = 0;
    int correct = 0;
    for (int i = 0; i < test_images->length; i++) {
        for (int j = 0; j < layers[0].prev_size; j++) {
            nn_input[j] = (float) test_images->data[i * layers[0].prev_size + j] / (float) 255;
        }

        forward_propagation(layers, NUM_LAYERS, nn_input);
        const float* out = get_layer_output(layers, NUM_LAYERS);

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

    end = clock();
    cpu_time = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Testing took: %f\n", cpu_time);
    
    printf("Total: %d\n", total);
    printf("Correct: %d\n", correct);
    printf("Accuracy: %f\n", (float) correct / (float) total);

    _mm_free(nn_input);
    free_layers(layers, NUM_LAYERS);
    free(labels);
    free_images(images);
    free(test_labels);
    free_images(test_images);
    
    return 0;
}
