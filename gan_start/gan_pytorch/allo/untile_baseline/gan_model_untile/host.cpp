//=============================================================================
// Host code for GAN model with all five GenBlocks and final ConvTranspose
//=============================================================================

// OpenCL utility layer include
#include "xcl2.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <vector>
#include <iomanip>
#include <fstream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string binaryFile = argv[1];
    cl_int err;
    cl::CommandQueue q;
    cl::Context context;
    cl::Program program;
    cl::Kernel krnl_top;

    // Define constants
    const int BATCH_SIZE = 8;
    const int KERNEL_SIZE = 4;
    
    // Block 0: 128 -> 1024, 1x1 -> 4x4
    const int IN_CHANNELS_0 = 128;
    const int OUT_CHANNELS_0 = 256;
    const int INPUT_HEIGHT_0 = 1;
    const int INPUT_WIDTH_0 = 1;
    
    // Block 1: 1024 -> 512, 4x4 -> 8x8
    const int IN_CHANNELS_1 = 256;
    const int OUT_CHANNELS_1 = 128;
    
    // Block 2: 512 -> 256, 8x8 -> 16x16
    const int IN_CHANNELS_2 = 128;
    const int OUT_CHANNELS_2 = 64;
    
    // Block 3: 256 -> 128, 16x16 -> 32x32
    const int IN_CHANNELS_3 = 64;
    const int OUT_CHANNELS_3 = 32;
    
    // Block 4: 128 -> 64, 32x32 -> 64x64
    const int IN_CHANNELS_4 = 32;
    const int OUT_CHANNELS_4 = 16;
    
    // Final layer: 64 -> 3, 64x64 -> 128x128
    const int IN_CHANNELS_5 = 16;
    const int OUT_CHANNELS_5 = 3;
    const int OUTPUT_HEIGHT = 128;
    const int OUTPUT_WIDTH = 128;
    
    // Calculate sizes for buffers
    const int input_size = BATCH_SIZE * IN_CHANNELS_0 * INPUT_HEIGHT_0 * INPUT_WIDTH_0;
    const int weight_size_0 = IN_CHANNELS_0 * OUT_CHANNELS_0 * KERNEL_SIZE * KERNEL_SIZE;
    const int bias_size_0 = OUT_CHANNELS_0;
    const int weight_size_1 = IN_CHANNELS_1 * OUT_CHANNELS_1 * KERNEL_SIZE * KERNEL_SIZE;
    const int bias_size_1 = OUT_CHANNELS_1;
    const int weight_size_2 = IN_CHANNELS_2 * OUT_CHANNELS_2 * KERNEL_SIZE * KERNEL_SIZE;
    const int bias_size_2 = OUT_CHANNELS_2;
    const int weight_size_3 = IN_CHANNELS_3 * OUT_CHANNELS_3 * KERNEL_SIZE * KERNEL_SIZE;
    const int bias_size_3 = OUT_CHANNELS_3;
    const int weight_size_4 = IN_CHANNELS_4 * OUT_CHANNELS_4 * KERNEL_SIZE * KERNEL_SIZE;
    const int bias_size_4 = OUT_CHANNELS_4;
    const int weight_size_5 = IN_CHANNELS_5 * OUT_CHANNELS_5 * KERNEL_SIZE * KERNEL_SIZE;
    const int bias_size_5 = OUT_CHANNELS_5;
    const int output_size = BATCH_SIZE * OUT_CHANNELS_5 * OUTPUT_HEIGHT * OUTPUT_WIDTH;

    // Calculate sizes for intermediate buffers
    const int block0_output_size = BATCH_SIZE * OUT_CHANNELS_0 * 4 * 4;
    const int block1_output_size = BATCH_SIZE * OUT_CHANNELS_1 * 8 * 8;
    const int block2_output_size = BATCH_SIZE * OUT_CHANNELS_2 * 16 * 16;
    const int block3_output_size = BATCH_SIZE * OUT_CHANNELS_3 * 32 * 32;
    const int block4_output_size = BATCH_SIZE * OUT_CHANNELS_4 * 64 * 64;

    // Allocate Memory in Host Memory
    // Read input noise vector
    std::ifstream ifile0("input0.data");
    if (!ifile0.is_open()) {
      std::cerr << "Error: Could not open input0.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_input(input_size);
    for (unsigned i = 0; i < input_size; i++) {
      ifile0 >> source_input[i];
    }
    ifile0.close();
    
    // Read block 0 parameters
    // Weights for block 0
    std::ifstream ifile1("input1.data");
    if (!ifile1.is_open()) {
      std::cerr << "Error: Could not open input1.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_weights_0(weight_size_0);
    for (unsigned i = 0; i < weight_size_0; i++) {
      ifile1 >> source_weights_0[i];
    }
    ifile1.close();
    
    // Bias for block 0
    std::ifstream ifile2("input2.data");
    if (!ifile2.is_open()) {
      std::cerr << "Error: Could not open input2.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_bias_0(bias_size_0);
    for (unsigned i = 0; i < bias_size_0; i++) {
      ifile2 >> source_bias_0[i];
    }
    ifile2.close();
    
    // BatchNorm parameters for block 0
    // Gamma
    std::ifstream ifile3("input3.data");
    if (!ifile3.is_open()) {
      std::cerr << "Error: Could not open input3.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_gamma_0(bias_size_0);
    for (unsigned i = 0; i < bias_size_0; i++) {
      ifile3 >> source_gamma_0[i];
    }
    ifile3.close();
    
    // Beta
    std::ifstream ifile4("input4.data");
    if (!ifile4.is_open()) {
      std::cerr << "Error: Could not open input4.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_beta_0(bias_size_0);
    for (unsigned i = 0; i < bias_size_0; i++) {
      ifile4 >> source_beta_0[i];
    }
    ifile4.close();
    
    // Running mean
    std::ifstream ifile5("input5.data");
    if (!ifile5.is_open()) {
      std::cerr << "Error: Could not open input5.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_mean_0(bias_size_0);
    for (unsigned i = 0; i < bias_size_0; i++) {
      ifile5 >> source_mean_0[i];
    }
    ifile5.close();
    
    // Running var
    std::ifstream ifile6("input6.data");
    if (!ifile6.is_open()) {
      std::cerr << "Error: Could not open input6.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_var_0(bias_size_0);
    for (unsigned i = 0; i < bias_size_0; i++) {
      ifile6 >> source_var_0[i];
    }
    ifile6.close();
    
    // Read block 1 parameters
    // Weights for block 1
    std::ifstream ifile7("input7.data");
    if (!ifile7.is_open()) {
      std::cerr << "Error: Could not open input7.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_weights_1(weight_size_1);
    for (unsigned i = 0; i < weight_size_1; i++) {
      ifile7 >> source_weights_1[i];
    }
    ifile7.close();
    
    // Bias for block 1
    std::ifstream ifile8("input8.data");
    if (!ifile8.is_open()) {
      std::cerr << "Error: Could not open input8.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_bias_1(bias_size_1);
    for (unsigned i = 0; i < bias_size_1; i++) {
      ifile8 >> source_bias_1[i];
    }
    ifile8.close();
    
    // BatchNorm parameters for block 1
    // Gamma
    std::ifstream ifile9("input9.data");
    if (!ifile9.is_open()) {
      std::cerr << "Error: Could not open input9.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_gamma_1(bias_size_1);
    for (unsigned i = 0; i < bias_size_1; i++) {
      ifile9 >> source_gamma_1[i];
    }
    ifile9.close();
    
    // Beta
    std::ifstream ifile10("input10.data");
    if (!ifile10.is_open()) {
      std::cerr << "Error: Could not open input10.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_beta_1(bias_size_1);
    for (unsigned i = 0; i < bias_size_1; i++) {
      ifile10 >> source_beta_1[i];
    }
    ifile10.close();
    
    // Running mean
    std::ifstream ifile11("input11.data");
    if (!ifile11.is_open()) {
      std::cerr << "Error: Could not open input11.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_mean_1(bias_size_1);
    for (unsigned i = 0; i < bias_size_1; i++) {
      ifile11 >> source_mean_1[i];
    }
    ifile11.close();
    
    // Running var
    std::ifstream ifile12("input12.data");
    if (!ifile12.is_open()) {
      std::cerr << "Error: Could not open input12.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_var_1(bias_size_1);
    for (unsigned i = 0; i < bias_size_1; i++) {
      ifile12 >> source_var_1[i];
    }
    ifile12.close();
    
    // Read block 2 parameters
    // Weights for block 2
    std::ifstream ifile13("input13.data");
    if (!ifile13.is_open()) {
      std::cerr << "Error: Could not open input13.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_weights_2(weight_size_2);
    for (unsigned i = 0; i < weight_size_2; i++) {
      ifile13 >> source_weights_2[i];
    }
    ifile13.close();
    
    // Bias for block 2
    std::ifstream ifile14("input14.data");
    if (!ifile14.is_open()) {
      std::cerr << "Error: Could not open input14.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_bias_2(bias_size_2);
    for (unsigned i = 0; i < bias_size_2; i++) {
      ifile14 >> source_bias_2[i];
    }
    ifile14.close();
    
    // BatchNorm parameters for block 2
    // Gamma
    std::ifstream ifile15("input15.data");
    if (!ifile15.is_open()) {
      std::cerr << "Error: Could not open input15.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_gamma_2(bias_size_2);
    for (unsigned i = 0; i < bias_size_2; i++) {
      ifile15 >> source_gamma_2[i];
    }
    ifile15.close();
    
    // Beta
    std::ifstream ifile16("input16.data");
    if (!ifile16.is_open()) {
      std::cerr << "Error: Could not open input16.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_beta_2(bias_size_2);
    for (unsigned i = 0; i < bias_size_2; i++) {
      ifile16 >> source_beta_2[i];
    }
    ifile16.close();
    
    // Running mean
    std::ifstream ifile17("input17.data");
    if (!ifile17.is_open()) {
      std::cerr << "Error: Could not open input17.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_mean_2(bias_size_2);
    for (unsigned i = 0; i < bias_size_2; i++) {
      ifile17 >> source_mean_2[i];
    }
    ifile17.close();
    
    // Running var
    std::ifstream ifile18("input18.data");
    if (!ifile18.is_open()) {
      std::cerr << "Error: Could not open input18.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_var_2(bias_size_2);
    for (unsigned i = 0; i < bias_size_2; i++) {
      ifile18 >> source_var_2[i];
    }
    ifile18.close();
    
    // Read block 3 parameters
    // Weights for block 3
    std::ifstream ifile19("input19.data");
    if (!ifile19.is_open()) {
      std::cerr << "Error: Could not open input19.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_weights_3(weight_size_3);
    for (unsigned i = 0; i < weight_size_3; i++) {
      ifile19 >> source_weights_3[i];
    }
    ifile19.close();
    
    // Bias for block 3
    std::ifstream ifile20("input20.data");
    if (!ifile20.is_open()) {
      std::cerr << "Error: Could not open input20.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_bias_3(bias_size_3);
    for (unsigned i = 0; i < bias_size_3; i++) {
      ifile20 >> source_bias_3[i];
    }
    ifile20.close();
    
    // BatchNorm parameters for block 3
    // Gamma
    std::ifstream ifile21("input21.data");
    if (!ifile21.is_open()) {
      std::cerr << "Error: Could not open input21.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_gamma_3(bias_size_3);
    for (unsigned i = 0; i < bias_size_3; i++) {
      ifile21 >> source_gamma_3[i];
    }
    ifile21.close();
    
    // Beta
    std::ifstream ifile22("input22.data");
    if (!ifile22.is_open()) {
      std::cerr << "Error: Could not open input22.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_beta_3(bias_size_3);
    for (unsigned i = 0; i < bias_size_3; i++) {
      ifile22 >> source_beta_3[i];
    }
    ifile22.close();
    
    // Running mean
    std::ifstream ifile23("input23.data");
    if (!ifile23.is_open()) {
      std::cerr << "Error: Could not open input23.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_mean_3(bias_size_3);
    for (unsigned i = 0; i < bias_size_3; i++) {
      ifile23 >> source_mean_3[i];
    }
    ifile23.close();
    
    // Running var
    std::ifstream ifile24("input24.data");
    if (!ifile24.is_open()) {
      std::cerr << "Error: Could not open input24.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_var_3(bias_size_3);
    for (unsigned i = 0; i < bias_size_3; i++) {
      ifile24 >> source_var_3[i];
    }
    ifile24.close();
    
    // Read block 4 parameters
    // Weights for block 4
    std::ifstream ifile25("input25.data");
    if (!ifile25.is_open()) {
      std::cerr << "Error: Could not open input25.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_weights_4(weight_size_4);
    for (unsigned i = 0; i < weight_size_4; i++) {
      ifile25 >> source_weights_4[i];
    }
    ifile25.close();
    
    // Bias for block 4
    std::ifstream ifile26("input26.data");
    if (!ifile26.is_open()) {
      std::cerr << "Error: Could not open input26.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_bias_4(bias_size_4);
    for (unsigned i = 0; i < bias_size_4; i++) {
      ifile26 >> source_bias_4[i];
    }
    ifile26.close();
    
    // BatchNorm parameters for block 4
    // Gamma
    std::ifstream ifile27("input27.data");
    if (!ifile27.is_open()) {
      std::cerr << "Error: Could not open input27.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_gamma_4(bias_size_4);
    for (unsigned i = 0; i < bias_size_4; i++) {
      ifile27 >> source_gamma_4[i];
    }
    ifile27.close();
    
    // Beta
    std::ifstream ifile28("input28.data");
    if (!ifile28.is_open()) {
      std::cerr << "Error: Could not open input28.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_beta_4(bias_size_4);
    for (unsigned i = 0; i < bias_size_4; i++) {
      ifile28 >> source_beta_4[i];
    }
    ifile28.close();
    
    // Running mean
    std::ifstream ifile29("input29.data");
    if (!ifile29.is_open()) {
      std::cerr << "Error: Could not open input29.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_mean_4(bias_size_4);
    for (unsigned i = 0; i < bias_size_4; i++) {
      ifile29 >> source_mean_4[i];
    }
    ifile29.close();
    
    // Running var
    std::ifstream ifile30("input30.data");
    if (!ifile30.is_open()) {
      std::cerr << "Error: Could not open input30.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_var_4(bias_size_4);
    for (unsigned i = 0; i < bias_size_4; i++) {
      ifile30 >> source_var_4[i];
    }
    ifile30.close();
    
    // Read final layer parameters
    // Weights for final layer
    std::ifstream ifile31("input31.data");
    if (!ifile31.is_open()) {
      std::cerr << "Error: Could not open input31.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_weights_5(weight_size_5);
    for (unsigned i = 0; i < weight_size_5; i++) {
      ifile31 >> source_weights_5[i];
    }
    ifile31.close();
    
    // Bias for final layer
    std::ifstream ifile32("input32.data");
    if (!ifile32.is_open()) {
      std::cerr << "Error: Could not open input32.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_bias_5(bias_size_5);
    for (unsigned i = 0; i < bias_size_5; i++) {
      ifile32 >> source_bias_5[i];
    }
    ifile32.close();
    
    // Output buffer initialization
    std::ifstream ifile33("input33.data");
    if (!ifile33.is_open()) {
      std::cerr << "Error: Could not open input33.data file.\n";
      return 1;
    }
    std::vector<float, aligned_allocator<float>> source_output(output_size);
    for (unsigned i = 0; i < output_size; i++) {
      ifile33 >> source_output[i];
    }
    ifile33.close();

    // OPENCL HOST CODE AREA START
    auto devices = xcl::get_xil_devices();
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;

    // Intermediate output buffers
    std::vector<float, aligned_allocator<float>> block0_output(block0_output_size, 0);
    std::vector<float, aligned_allocator<float>> block1_output(block1_output_size, 0);
    std::vector<float, aligned_allocator<float>> block2_output(block2_output_size, 0);
    std::vector<float, aligned_allocator<float>> block3_output(block3_output_size, 0);
    std::vector<float, aligned_allocator<float>> block4_output(block4_output_size, 0);
    
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_top = cl::Kernel(program, "top", &err));
            valid_device = true;
            break;
        }
    }
    
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
    
    // Allocate Buffers in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and Device-to-host communication
    OCL_CHECK(err, cl::Buffer buffer_input(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * input_size, source_input.data(), &err));
    
    // Block 0 buffers
    OCL_CHECK(err, cl::Buffer buffer_weights_0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                             sizeof(float) * weight_size_0, source_weights_0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_bias_0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_0, source_bias_0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_gamma_0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                           sizeof(float) * bias_size_0, source_gamma_0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_beta_0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_0, source_beta_0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_mean_0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_0, source_mean_0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_var_0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                         sizeof(float) * bias_size_0, source_var_0.data(), &err));
    
    // Block 1 buffers
    OCL_CHECK(err, cl::Buffer buffer_weights_1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                             sizeof(float) * weight_size_1, source_weights_1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_bias_1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_1, source_bias_1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_gamma_1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                           sizeof(float) * bias_size_1, source_gamma_1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_beta_1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_1, source_beta_1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_mean_1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_1, source_mean_1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_var_1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                         sizeof(float) * bias_size_1, source_var_1.data(), &err));
    
    // Block 2 buffers
    OCL_CHECK(err, cl::Buffer buffer_weights_2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                             sizeof(float) * weight_size_2, source_weights_2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_bias_2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_2, source_bias_2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_gamma_2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                           sizeof(float) * bias_size_2, source_gamma_2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_beta_2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_2, source_beta_2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_mean_2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_2, source_mean_2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_var_2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                         sizeof(float) * bias_size_2, source_var_2.data(), &err));
    
    // Block 3 buffers
    OCL_CHECK(err, cl::Buffer buffer_weights_3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                             sizeof(float) * weight_size_3, source_weights_3.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_bias_3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_3, source_bias_3.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_gamma_3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                           sizeof(float) * bias_size_3, source_gamma_3.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_beta_3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_3, source_beta_3.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_mean_3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                          sizeof(float) * bias_size_3, source_mean_3.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_var_3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                                         sizeof(float) * bias_size_3, source_var_3.data(), &err));
    
// Block 4 buffers (continued)
OCL_CHECK(err, cl::Buffer buffer_weights_4(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
    sizeof(float) * weight_size_4, source_weights_4.data(), &err));
OCL_CHECK(err, cl::Buffer buffer_bias_4(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
 sizeof(float) * bias_size_4, source_bias_4.data(), &err));
OCL_CHECK(err, cl::Buffer buffer_gamma_4(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
  sizeof(float) * bias_size_4, source_gamma_4.data(), &err));
OCL_CHECK(err, cl::Buffer buffer_beta_4(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
 sizeof(float) * bias_size_4, source_beta_4.data(), &err));
OCL_CHECK(err, cl::Buffer buffer_mean_4(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
 sizeof(float) * bias_size_4, source_mean_4.data(), &err));
OCL_CHECK(err, cl::Buffer buffer_var_4(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
sizeof(float) * bias_size_4, source_var_4.data(), &err));

// Final layer buffers
OCL_CHECK(err, cl::Buffer buffer_weights_5(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
    sizeof(float) * weight_size_5, source_weights_5.data(), &err));
OCL_CHECK(err, cl::Buffer buffer_bias_5(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
 sizeof(float) * bias_size_5, source_bias_5.data(), &err));

OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, 
  sizeof(float) * output_size, source_output.data(), &err));

  // Intermediate output buffers
OCL_CHECK(err, cl::Buffer buffer_block0_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 
  sizeof(float) * block0_output_size, block0_output.data(), &err));
OCL_CHECK(err, cl::Buffer buffer_block1_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 
  sizeof(float) * block1_output_size, block1_output.data(), &err));
OCL_CHECK(err, cl::Buffer buffer_block2_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 
  sizeof(float) * block2_output_size, block2_output.data(), &err));
OCL_CHECK(err, cl::Buffer buffer_block3_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 
  sizeof(float) * block3_output_size, block3_output.data(), &err));
OCL_CHECK(err, cl::Buffer buffer_block4_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, 
  sizeof(float) * block4_output_size, block4_output.data(), &err));

// Set kernel arguments
int arg_idx = 0;
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_input));  // Input noise vector

// Block 0 parameters
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_weights_0));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_bias_0));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_gamma_0));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_beta_0));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_mean_0));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_var_0));

// Block 1 parameters
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_weights_1));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_bias_1));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_gamma_1));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_beta_1));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_mean_1));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_var_1));

// Block 2 parameters
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_weights_2));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_bias_2));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_gamma_2));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_beta_2));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_mean_2));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_var_2));

// Block 3 parameters
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_weights_3));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_bias_3));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_gamma_3));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_beta_3));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_mean_3));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_var_3));

// Block 4 parameters
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_weights_4));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_bias_4));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_gamma_4));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_beta_4));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_mean_4));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_var_4));

// Final layer parameters (no BatchNorm)
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_weights_5));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_bias_5));

// Intermediate output buffers
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_block0_output));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_block1_output));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_block2_output));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_block3_output));
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_block4_output));
// Output
OCL_CHECK(err, err = krnl_top.setArg(arg_idx++, buffer_output));

// Copy input data to device global memory
std::vector<cl::Memory> inBufVec;
inBufVec.push_back(buffer_input);

// Block 0
inBufVec.push_back(buffer_weights_0);
inBufVec.push_back(buffer_bias_0);
inBufVec.push_back(buffer_gamma_0);
inBufVec.push_back(buffer_beta_0);
inBufVec.push_back(buffer_mean_0);
inBufVec.push_back(buffer_var_0);

// Block 1
inBufVec.push_back(buffer_weights_1);
inBufVec.push_back(buffer_bias_1);
inBufVec.push_back(buffer_gamma_1);
inBufVec.push_back(buffer_beta_1);
inBufVec.push_back(buffer_mean_1);
inBufVec.push_back(buffer_var_1);

// Block 2
inBufVec.push_back(buffer_weights_2);
inBufVec.push_back(buffer_bias_2);
inBufVec.push_back(buffer_gamma_2);
inBufVec.push_back(buffer_beta_2);
inBufVec.push_back(buffer_mean_2);
inBufVec.push_back(buffer_var_2);

// Block 3
inBufVec.push_back(buffer_weights_3);
inBufVec.push_back(buffer_bias_3);
inBufVec.push_back(buffer_gamma_3);
inBufVec.push_back(buffer_beta_3);
inBufVec.push_back(buffer_mean_3);
inBufVec.push_back(buffer_var_3);

// Block 4
inBufVec.push_back(buffer_weights_4);
inBufVec.push_back(buffer_bias_4);
inBufVec.push_back(buffer_gamma_4);
inBufVec.push_back(buffer_beta_4);
inBufVec.push_back(buffer_mean_4);
inBufVec.push_back(buffer_var_4);

// Final layer
inBufVec.push_back(buffer_weights_5);
inBufVec.push_back(buffer_bias_5);

// Output

// Intermediate output buffers
inBufVec.push_back(buffer_block0_output);
inBufVec.push_back(buffer_block1_output);
inBufVec.push_back(buffer_block2_output);
inBufVec.push_back(buffer_block3_output);
inBufVec.push_back(buffer_block4_output);

inBufVec.push_back(buffer_output);

OCL_CHECK(err, err = q.enqueueMigrateMemObjects(inBufVec, 0 /* 0 means from host*/));

cl::Event event;
uint64_t nstimestart, nstimeend;
std::cout << "|-------------------------+-------------------------|\n"
<< "| Kernel                  |    Wall-Clock Time (ns) |\n"
<< "|-------------------------+-------------------------|\n";

// Launch the Kernel
OCL_CHECK(err, err = q.enqueueTask(krnl_top, nullptr, &event));

// Copy Result from Device Global Memory to Host Local Memory
OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
q.finish();

// Get the execution time
OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
auto exe_time = nstimeend - nstimestart;

std::cout << "| " << std::left << std::setw(24) << "top"
<< "|" << std::right << std::setw(24) << exe_time << " |\n";
std::cout << "|-------------------------+-------------------------|\n";
std::cout << "Note: Wall Clock Time is meaningful for real hardware execution "
<< "only, not for emulation.\n";
std::cout << "Please refer to profile summary for kernel execution time for "
<< "hardware emulation.\n";
std::cout << "Finished execution!\n\n";

// Write the output data to file
std::ofstream ofile;
ofile.open("output.data");
if (!ofile) {
std::cerr << "Failed to open output file!" << std::endl;
return EXIT_FAILURE;
}
for (unsigned i = 0; i < output_size; i++) {
ofile << source_output[i] << std::endl;
}
ofile.close();

std::cout << "Output size: " << output_size << " elements written to output.data" << std::endl;
std::cout << "Output dimensions: " << BATCH_SIZE << "x" << OUT_CHANNELS_5 << "x" 
<< OUTPUT_HEIGHT << "x" << OUTPUT_WIDTH << std::endl;
return EXIT_SUCCESS;
}