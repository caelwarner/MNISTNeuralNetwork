#ifndef MAIN_H
#define MAIN_H

#include <stdlib.h>

#define NUM_OUTPUTS 10
static const int LAYER_SIZES[] = {128, 64, NUM_OUTPUTS};
#define NUM_LAYERS (sizeof(LAYER_SIZES) / sizeof(int))

#define NUM_EPOCHS 10
#define LEARNING_RATE 0.001f

typedef struct {
    int size;
    int prev_size;
    float* weights;
    float* biases;
    float* activation;
} NNLayer;

static float rand_float() {
    return (float) rand() / (float) RAND_MAX;
}

float he_uniform_init(int fan_in);

NNLayer* create_layers(int input_layer_size, const int* layer_sizes, int num_layers);
void free_layers(NNLayer* layers, int num_layers);
float* get_layer_output(const NNLayer* layer, int num_layers);

void generate_expected_vec(float* expected_vec, int expected);

void forward_propagation(NNLayer* layers, int num_layers, const float* input);
void backward_propagation(NNLayer* layers, int num_layers, int max_layer_size, const float* input, const float* expected);
float cost(const float* nn_output, const float* expected);

#endif //MAIN_H
