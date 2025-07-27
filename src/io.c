#include "io.h"


void free_images(Images *images) {
    free(images->data);
    free(images);
}

Images* read_images(const char* filename) {
    FILE* file = fopen(filename, "rb");
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

    if (magic_number != 2051) {
        fprintf(stderr, "Invalid magic number!\n");
        return NULL;
    }
    
    uint8_t* data = malloc(length * rows * cols * sizeof(uint8_t));
    fread(data, sizeof(uint8_t), length * rows * cols, file);

    Images* images = malloc(sizeof(Images));
    images->data = data;
    images->rows = rows;
    images->cols = cols;
    images->length = length;

    fclose(file);
    
    return images;
}

uint8_t * read_labels(const char* filename) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Error opening file!\n");
        return NULL;
    }

    uint8_t header[8];
    fread(header, sizeof(int), 2, file);

    const int magic_number = flip_endianness(header);
    const int n = flip_endianness(header + 4);

    if (magic_number != 2049) {
        fprintf(stderr, "Invalid magic number!\n");
        return NULL;
    }
    
    uint8_t* labels = malloc(n * sizeof(uint8_t));
    fread(labels, sizeof(uint8_t), n, file);

    fclose(file);
    
    return labels;
}
