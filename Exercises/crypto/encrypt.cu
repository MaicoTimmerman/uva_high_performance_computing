#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "timer.h"

using namespace std;

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
 */
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}


__global__ void encryptKernel(int key_size, char* deviceDataIn, short* deviceDataKey, char* deviceDataOut) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned key_index = index % key_size;
    deviceDataOut[index] = deviceDataIn[index] + deviceDataKey[key_index];
}

__global__ void decryptKernel(int key_size, char* deviceDataIn, short* deviceDataKey, char* deviceDataOut) {
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned key_index = index % key_size;
    deviceDataOut[index] = deviceDataIn[index] - deviceDataKey[key_index];
}

int fileSize(const char *fn) {
    int size;

    ifstream file (fn, ios::in|ios::binary|ios::ate);
    if (file.is_open())
    {
        size = file.tellg();
        file.close();
    }
    else {
        cout << "Unable to open file";
        size = -1;
    }
    return size;
}

int readData(string fileName, char *data) {

    streampos size;

    ifstream file (fileName, ios::in|ios::binary|ios::ate);
    if (file.is_open())
    {
        size = file.tellg();
        file.seekg (0, ios::beg);
        file.read (data, size);
        file.close();

        cout << "The entire file content is in memory." << endl;
    }
    else cout << "Unable to open file" << endl;
    return 0;
}

int writeData(int size, string fileName, char *data) {
    ofstream file (fileName, ios::out|ios::binary|ios::trunc);
    if (file.is_open())
    {
        file.write (data, size);
        file.close();

        cout << "The entire file content was written to file." << endl;
        return 0;
    }
    else cout << "Unable to open file";

    return -1;
}

int EncryptSeq (int n, int key_size, char* data_in, short* data_key, char* data_out)
{
    timer sequentialTime = timer("Sequential encryption");

    sequentialTime.start();
    int key_index = 0;
    for (int i=0; i<n; i++) {
        data_out[i] = data_in[i] + data_key[key_index];
        key_index %= key_size;
    }
    sequentialTime.stop();

    cout << fixed << setprecision(6);
    cout << "Encryption (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;

    return 0;
}

int DecryptSeq (int n, int key_size, char* data_in, short* data_key, char* data_out)
{
    timer sequentialTime = timer("Sequential decryption");

    sequentialTime.start();
    int key_index = 0;
    for (int i=0; i<n; i++) {
        data_out[i] = data_in[i] - data_key[key_index];
        key_index %= key_size;
    }
    sequentialTime.stop();

    cout << fixed << setprecision(6);
    cout << "Decryption (sequential): \t\t" << sequentialTime.getElapsed() << " seconds." << endl;

    return 0;
}


int EncryptCuda (int n, int key_size, char* data_in, short* data_key, char* data_out) {
    int threadBlockSize = 512;

    // allocate the vectors on the GPU
    char* deviceDataIn = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataIn, n * sizeof(char)));
    if (deviceDataIn == NULL) {
        cout << "could not allocate memory!" << endl;
        return -1;
    }
    char* deviceDataOut = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataOut, n * sizeof(char)));
    if (deviceDataOut == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        cout << "could not allocate memory!" << endl;
        return -1;
    }

    short* deviceDataKey = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataKey, key_size * sizeof(short)));
    if (deviceDataOut == NULL) {
        checkCudaCall(cudaFree(deviceDataOut));
        checkCudaCall(cudaFree(deviceDataIn));
        cout << "could not allocate memory!" << endl;
        return -1;
    }

    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(char), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceDataKey, data_key, key_size*sizeof(short), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    kernelTime1.start();
    encryptKernel<<<n/threadBlockSize+1, threadBlockSize>>>(key_size, deviceDataIn, deviceDataKey, deviceDataOut);
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(data_out, deviceDataOut, n * sizeof(char), cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(deviceDataIn));
    checkCudaCall(cudaFree(deviceDataOut));
    checkCudaCall(cudaFree(deviceDataKey));
    memoryTime.stop();

    cout << fixed << setprecision(6);
    cout << "Encrypt (kernel): \t\t" << kernelTime1.getElapsed() << " seconds." << endl;
    cout << "Encrypt (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;

    return 0;
}

int DecryptCuda (int n, int key_size, char* data_in, short* data_key, char* data_out) {
    int threadBlockSize = 512;

    // allocate the vectors on the GPU
    char* deviceDataIn = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataIn, n * sizeof(char)));
    if (deviceDataIn == NULL) {
        cout << "could not allocate memory!" << endl;
        return -1;
    }
    char* deviceDataOut = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataOut, n * sizeof(char)));
    if (deviceDataOut == NULL) {
        checkCudaCall(cudaFree(deviceDataIn));
        cout << "could not allocate memory!" << endl;
        return -1;
    }

    short* deviceDataKey = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceDataKey, key_size * sizeof(short)));
    if (deviceDataOut == NULL) {
        checkCudaCall(cudaFree(deviceDataOut));
        checkCudaCall(cudaFree(deviceDataIn));
        cout << "could not allocate memory!" << endl;
        return -1;
    }

    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceDataIn, data_in, n*sizeof(char), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceDataKey, data_key, key_size*sizeof(short), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    kernelTime1.start();
    decryptKernel<<<n/threadBlockSize + 1, threadBlockSize>>>(key_size, deviceDataIn, deviceDataKey, deviceDataOut);
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(data_out, deviceDataOut, n * sizeof(char), cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(deviceDataIn));
    checkCudaCall(cudaFree(deviceDataOut));
    memoryTime.stop();

    cout << fixed << setprecision(6);
    cout << "Decrypt (kernel): \t\t" << kernelTime1.getElapsed() << " seconds." << endl;
    cout << "Decrypt (memory): \t\t" << memoryTime.getElapsed() << " seconds." << endl;

    return 0;
}

int main(int argc, char* argv[]) {
    int data_size, key_size;

    data_size = fileSize("original.data");
    key_size = fileSize("cipher.key");
    if (data_size == -1) {
        cout << "Original file not found! Exiting ... " << endl;
        exit(0);
    }
    if (key_size == -1) {
        cout << "Original file not found! Exiting ... " << endl;
        exit(0);
    }

    char* data_in = new char[data_size];
    char* data_out = new char[data_size];
    char* raw_data_key = new char[key_size];
    short* data_key = new short[key_size];

    readData("original.data", data_in);
    readData("cipher.key", raw_data_key);

    // Check if the key is numerical.
    for (int i=0; i<key_size; i++) {
        if (raw_data_key[i] < '0' || raw_data_key[i] > '9') {
            cout << "Key contains non-numerical value: " << raw_data_key[i] << "! Exiting ..." << endl;
            exit(0);
        }

        data_key[i] = raw_data_key[i] - '0';
    }

    cout << "Using '";
    for (int i=0; i<key_size; i++) {
        cout << data_key[i];
    }
    cout << "' as cipher key." << endl;
    cout << "Encrypting a file of " << data_size << " characters." << endl;

    EncryptSeq(data_size, key_size, data_in, data_key, data_out);
    writeData(data_size, "sequential.data", data_out);

    EncryptCuda(data_size, key_size, data_in, data_key, data_out);
    writeData(data_size, "cuda.data", data_out);

    readData("cuda.data", data_in);
    readData("sequential.data", data_in);

    cout << "Decrypting a file of " << data_size << "characters" << endl;
    DecryptSeq(data_size, key_size, data_in, data_key, data_out);
    writeData(data_size, "sequential_decrypted.data", data_out);
    DecryptCuda(data_size, key_size, data_in, data_key, data_out);
    writeData(data_size, "recovered.data", data_out);

    delete[] data_in;
    delete[] data_out;
    delete[] data_key;

    return 0;
}
