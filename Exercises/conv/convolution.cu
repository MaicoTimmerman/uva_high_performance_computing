#include <stdio.h>

extern "C" {
    void start_timer();
    void stop_timer(float *time);
    __global__ void convolution_kernel(float *output, float *input, float *filter);
    __global__ void convolution_kernel_naive(float *output, float *input, float *filter);
}

int compare_arrays(float *c, float *d, int n);
void print_matrix(float *c, int d, int n);


#define image_height 1024
#define image_width 1024
#define filter_height 17
#define filter_width 17

#define block_size_x 32
#define block_size_y 16

#define border_height ((filter_height/2)*2)
#define border_width ((filter_width/2)*2)
#define input_height (image_height + border_height)
#define input_width (image_width + border_width)


void convolve(float *output, float *input, float *filter) {
    //for each pixel in the output image
    for (int y=0; y < image_height; y++) {
        for (int x=0; x < image_width; x++) {

            //for each filter weight
            for (int i=0; i < filter_height; i++) {
                for (int j=0; j < filter_width; j++) {
                    output[y*image_width+x] += input[(y+i)*input_width+x+j] * filter[i*filter_width+j];
                }
            }

        }
    }

}


__global__ void convolution_kernel_naive(float *output, float *input, float *filter) {
    //for each pixel in the output image
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    //thread-local register to hold local sum
    float sum = 0.0f;

    //for each filter weight
    for (int i=0; i < filter_height; i++) {
        for (int j=0; j < filter_width; j++) {
            sum += input[(y+i)*input_width+x+j] * filter[i*filter_width+j];
        }
    }

    //store result to global memory
    output[y * image_width + x] = sum;

}

// i = x = height
// j = y = width

#define block_mem_width (block_size_x + filter_width - 1)
#define block_mem_height (block_size_y + filter_height - 1)


__global__ void convolution_kernel(float *output, float *input, float *filter, unsigned int *counter) {
    //declare shared memory for this thread block
    //the area reserved is equal to the thread block size plus
    //the size of the border needed for the computation

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bdx = blockDim.x;
    int bdy = blockDim.y;

    // coordinates in block image. intentional integer division
    int block_img_row = by * (bdy - 2 * (filter_height / 2)) + ty;
    int block_img_col = bx * (bdx - 2 * (filter_width / 2)) + tx;

    // coordinates in the real image.
    int input_row = block_img_row - (filter_height / 2);
    int input_col = block_img_col - (filter_width / 2);


    bool is_thread_vertical_in_block_image = block_img_row < image_height + filter_height;
    bool is_thread_horizontal_in_block_image = block_img_col < image_width + filter_width;

    bool is_within_padding = is_thread_vertical_in_block_image && is_thread_horizontal_in_block_image;
    bool is_within_image = input_row >=0 && input_col >= 0 && input_row < image_height && input_col < image_width;

    // make sure thread is within augmented boundaries
    if (!is_within_padding) {
        return;
    }


    __shared__ float shared_mem[block_mem_width][block_mem_height];

    if (is_within_image) {
        shared_mem[ty][tx] = input[(input_row * image_width + input_col)];
    }
    else {
        shared_mem[ty][tx] = 0;
    }


    // synchronize to make all writes visible to all threads within the thread block
    __syncthreads();

    float sum = 0.0f;

    int filter_row, filter_col;

    int filter_radius_width = filter_width / 2;
    int filter_radius_height = filter_height / 2;

    bool is_in_local_image = (tx >= filter_radius_width) && (ty >= filter_radius_height) && (ty < bdy - filter_radius_height) && (tx < bdx - filter_radius_width);

    if (is_within_image && is_in_local_image) {
        atomicAdd(counter, 1);

        //compute using shared memory
        //for each filter weight
        for (int i=0; i < filter_width; i++) {
            for (int j=0; j < filter_height; j++) {
                filter_col = tx + i;
                filter_row = ty + j;
                sum += shared_mem[filter_row][filter_col] * filter[j*filter_width+i];
            }
        }

        // store result to global memory
        output[input_row * image_width + input_col] = sum;
    }
}

void print_matrix(float *image, int width, int height) {
    for (int row=0; row <height; row++) {
        for(int columns=0; columns<width; columns++) {
            printf("%01d ", int(image[row * width + columns]));
        }
        printf("\n");
    }
}

int main() {

    float time;
    cudaError_t err;
    int errors = 0;

    //allocate arrays and fill them
    float *input = (float *) malloc(input_height * input_width * sizeof(float));
    float *output1 = (float *) malloc(image_height * image_width * sizeof(float));
    float *output2 = (float *) malloc(image_height * image_width * sizeof(float));
    float *filter = (float *) malloc(filter_height * filter_width * sizeof(float));
    for (int i=0; i< input_height * input_width; i++) {
        /* input[i] = 1.0 / rand(); */
        input[i] = 0;
    }

    /* input[int(ceil(image_height / 2) * image_width + ceil(image_width / 2))] = 1; */
    input[1*image_width+1] = 1;
    for (int i=0; i< filter_height * filter_width; i++) {
        input[i] = 1.0 / rand();
        /* filter[i] = 1; */
    }

    /* printf("Image\n"); */
    /* print_matrix(input, image_width, image_height); */
    /* printf("Filter\n"); */
    /* print_matrix(filter, filter_width, filter_height); */

    memset(output1, 0, image_height * image_width * sizeof(float));
    memset(output2, 0, image_height * image_width * sizeof(float));

    //measure the CPU function
    start_timer();
    convolve(output1, input, filter);
    stop_timer(&time);
    printf("convolution sequential took \t\t %.3f ms\n", time);

    /* printf("Output sequential\n"); */
    /* print_matrix(output1, image_width, image_height); */

    //allocate GPU memory
    float *d_input; float *d_output; float *d_filter;
    err = cudaMalloc((void **)&d_input, input_height*input_width*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMalloc d_input: %s\n", cudaGetErrorString( err ));
        errors++;
    }
    err = cudaMalloc((void **)&d_output, image_height*image_width*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMalloc d_output: %s\n", cudaGetErrorString( err ));
        errors++;
    }
    err = cudaMalloc((void **)&d_filter, filter_height*filter_width*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMalloc d_filter: %s\n", cudaGetErrorString( err ));
        errors++;
    }

    //copy the input data to the GPU
    err = cudaMemcpy(d_input, input, input_height*input_width*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpy host to device input: %s\n", cudaGetErrorString( err ));
        errors++;
    }
    err = cudaMemcpy(d_filter, filter, filter_height*filter_width*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpy host to device filter: %s\n", cudaGetErrorString( err ));
        errors++;
    }

    //zero the output array
    err = cudaMemset(d_output, 0, image_height*image_width*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemset output: %s\n", cudaGetErrorString( err ));
        errors++;
    }

    //setup the grid and thread blocks
    //thread block size
    dim3 threads(block_size_x, block_size_y);
    //problem size divided by thread block size rounded up
    dim3 grid(int(ceilf(image_width/(float)threads.x)),
            int(ceilf(image_height/(float)threads.y)) );

    //measure the GPU function
    cudaDeviceSynchronize();
    start_timer();
    convolution_kernel_naive<<<grid, threads>>>(d_output, d_input, d_filter);
    cudaDeviceSynchronize();
    stop_timer(&time);
    printf("convolution_kernel_naive took \t\t %.3f ms\n", time);

    //check to see if all went well
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error during kernel launch convolution_kernel: %s\n", cudaGetErrorString( err ));
        errors++;
    }

    //copy the result back to host memory
    err = cudaMemcpy(output2, d_output, image_height*image_width*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpy device to host output: %s\n", cudaGetErrorString( err ));
        errors++;
    }

    //check the result
    errors += compare_arrays(output1, output2, image_height*image_width);
    if (errors > 0) {
        printf("TEST FAILED! %d errors!\n", errors);
    } else {
        printf("TEST PASSED!\n");
    }

    //zero the output arrays
    errors = 0;
    memset(output2, 0, image_height*image_width*sizeof(float));
    err = cudaMemset(d_output, 0, image_height*image_width*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemset output: %s\n", cudaGetErrorString( err ));
        errors++;
    }

    //measure the GPU function
    unsigned int counter = 0;
    unsigned int *d_counter;
    err = cudaMalloc((void **)&d_counter, sizeof(unsigned int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error during malloc counter: %s\n", cudaGetErrorString( err ));
        errors++;
    }
    err = cudaMemcpy(d_counter, &counter, sizeof(unsigned int), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error during memcpy counter: %s\n", cudaGetErrorString( err ));
        errors++;
    }

    dim3 threads2(block_size_x, block_size_y);
    //problem size divided by thread block size rounded up
    /* printf("blocksize x %d\n", block_size_x); */
    /* printf("blocksize y %d\n", block_size_y); */
    /* printf("denom x %f\n", ((float)threads.x - filter_width - 1)); */
    /* printf("denom y %f\n", ((float)threads.y - filter_height - 1)); */
    /* printf("grid x %d\n", int(ceilf(image_width/((float)threads.x - (filter_width / 2) - 1)))); */
    /* printf("grid y %d\n", int(ceilf(image_height/((float)threads.y - (filter_height / 2) - 1)))); */
    dim3 grid2(int(ceilf(image_width/((float)threads.x - (filter_width / 2) - 1))),
            int(ceilf(image_height/((float)threads.y - (filter_height / 2) - 1))));

    start_timer();
    convolution_kernel<<<grid2, threads2>>>(d_output, d_input, d_filter, d_counter);
    cudaDeviceSynchronize();
    stop_timer(&time);
    printf("convolution_kernel_shared_mem took \t\t %.3f ms\n", time);

    //check to see if all went well
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error during kernel launch convolution_kernel_shared_mem: %s\n", cudaGetErrorString( err ));
        errors++;
    }

    //copy the result back to host memory
    err = cudaMemcpy(output2, d_output, image_height*image_width*sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpy device to host output: %s\n", cudaGetErrorString( err ));
        errors++;
    }

    err = cudaMemcpy(&counter, d_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaMemcpy device to host counter: %s\n", cudaGetErrorString( err ));
        errors++;
    }

    printf("convolution_kernel_shared_mem coutner \t\t %d\n", counter);
    printf("output shared memory\n");
    /* print_matrix(output2, image_width, image_height); */
    //debug
    /* for (int i = 0; i < image_height*image_width; ++i) { */
    /*     if (output2[i] > 0.0f) */
    /*         printf("%10.7e", output2[i]); */
    /* } */

    //check the result
    errors += compare_arrays(output1, output2, image_height*image_width);
    if (errors > 0) {
        printf("TEST FAILED! %d errors!\n", errors);
    } else {
        printf("TEST PASSED!\n");
    }

    //clean up
    cudaFree(d_output);
    cudaFree(d_input);
    cudaFree(d_filter);
    free(filter);
    free(input);
    free(output1);
    free(output2);

    return 0;
}

int compare_arrays(float *a1, float *a2, int n) {
    int errors = 0;
    int print = 0;

    for (int i=0; i<n; i++) {

        if (isnan(a1[i]) || isnan(a2[i])) {
            errors++;
            if (print < 10) {
                print++;
                fprintf(stderr, "Error NaN detected at i=%d,\t a1= %10.7e \t a2= \t %10.7e\n",i,a1[i],a2[i]);
            }
        }

        float diff = (a1[i]-a2[i])/a1[i];
        if (diff > 1e-6f) {
            errors++;
            if (print < 10) {
                print++;
                fprintf(stderr, "Error detected at i=%d, \t a1= \t %10.7e \t a2= \t %10.7e \t rel_error=\t %10.7e\n",i,a1[i],a2[i],diff);
            }
        }

    }


    return errors;
}
