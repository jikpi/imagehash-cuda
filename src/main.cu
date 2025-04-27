#include <iostream>

#include "constants_common_decl.h"
#include "constants_device_decl.h"
#include "constants_host_decl.h"
#include "../common/pg/cudaDefs.h"

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

int main(int argc, char *argv[]) {
    initializeCUDA(deviceProp);

    std::cout << "Hello, World!\n" << std::endl << TEST << THREADS_PER_BLOCK << std::endl
    //does nothing, testing purposes
    << KERNEL_DATA_PATH << std::endl;
}
