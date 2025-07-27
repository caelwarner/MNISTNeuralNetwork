#ifndef IO_H
#define IO_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int flip_endianness(const uint8_t* input) {
    return input[0] << 24 | input[1] << 16 | input[2] << 8 | input[3];
}

typedef struct {
    uint8_t* data;
    int rows;
    int cols;
    int length;
} Images;

void free_images(Images* images);

Images* read_images(const char* filename);

uint8_t* read_labels(const char* filename);

#endif //IO_H
