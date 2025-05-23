//=============================================================================
// Auto generated by Allo
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
    const int BATCH_SIZE = 4;
    const int IN_CHANNELS = 3;
    const int OUT_CHANNELS = 16;
    const int INPUT_HEIGHT = 64;
    const int INPUT_WIDTH = 64;
    const int KERNEL_SIZE = 4;
    const int STRIDE = 2;
    const int PADDING = 0;
    const int OUTPUT_HEIGHT = STRIDE * (INPUT_HEIGHT - 1) + KERNEL_SIZE - 2 * PADDING; // 130
    const int OUTPUT_WIDTH = STRIDE * (INPUT_WIDTH - 1) + KERNEL_SIZE - 2 * PADDING;   // 130
    
    const int input_size = BATCH_SIZE * IN_CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;      // 49,152
    const int weight_size = IN_CHANNELS * OUT_CHANNELS * KERNEL_SIZE * KERNEL_SIZE;    // 768
    const int bias_size = OUT_CHANNELS;                                                // 16
    const int output_size = BATCH_SIZE * OUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH;  // 1,081,600

    // Allocate Memory in Host Memory
    // Read input tensor
    std::ifstream ifile0("input0.data");
    if (!ifile0.is_open()) {
      std::cerr << "Error: Could not open input0.data file.\n";
      return 1;
    }
    float in_data_0[input_size];
    for (unsigned i = 0; i < input_size; i++) {
      ifile0 >> in_data_0[i];
    }
    size_t size_bytes_in0 = sizeof(float) * input_size;
    std::vector<float, aligned_allocator<float>> source_in0(in_data_0, in_data_0 + input_size);
    
    // Read convolution weights
    std::ifstream ifile1("input1.data");
    if (!ifile1.is_open()) {
      std::cerr << "Error: Could not open input1.data file.\n";
      return 1;
    }
    float in_data_1[weight_size];
    for (unsigned i = 0; i < weight_size; i++) {
      ifile1 >> in_data_1[i];
    }
    size_t size_bytes_in1 = sizeof(float) * weight_size;
    std::vector<float, aligned_allocator<float>> source_in1(in_data_1, in_data_1 + weight_size);
    
    // Read convolution bias
    std::ifstream ifile2("input2.data");
    if (!ifile2.is_open()) {
      std::cerr << "Error: Could not open input2.data file.\n";
      return 1;
    }
    float in_data_2[bias_size];
    for (unsigned i = 0; i < bias_size; i++) {
      ifile2 >> in_data_2[i];
    }
    size_t size_bytes_in2 = sizeof(float) * bias_size;
    std::vector<float, aligned_allocator<float>> source_in2(in_data_2, in_data_2 + bias_size);
    
    // Read batchnorm gamma
    std::ifstream ifile3("input3.data");
    if (!ifile3.is_open()) {
      std::cerr << "Error: Could not open input3.data file.\n";
      return 1;
    }
    float in_data_3[bias_size];
    for (unsigned i = 0; i < bias_size; i++) {
      ifile3 >> in_data_3[i];
    }
    size_t size_bytes_in3 = sizeof(float) * bias_size;
    std::vector<float, aligned_allocator<float>> source_in3(in_data_3, in_data_3 + bias_size);
    
    // Read batchnorm beta
    std::ifstream ifile4("input4.data");
    if (!ifile4.is_open()) {
      std::cerr << "Error: Could not open input4.data file.\n";
      return 1;
    }
    float in_data_4[bias_size];
    for (unsigned i = 0; i < bias_size; i++) {
      ifile4 >> in_data_4[i];
    }
    size_t size_bytes_in4 = sizeof(float) * bias_size;
    std::vector<float, aligned_allocator<float>> source_in4(in_data_4, in_data_4 + bias_size);
    
    // Read batchnorm running_mean
    std::ifstream ifile5("input5.data");
    if (!ifile5.is_open()) {
      std::cerr << "Error: Could not open input5.data file.\n";
      return 1;
    }
    float in_data_5[bias_size];
    for (unsigned i = 0; i < bias_size; i++) {
      ifile5 >> in_data_5[i];
    }
    size_t size_bytes_in5 = sizeof(float) * bias_size;
    std::vector<float, aligned_allocator<float>> source_in5(in_data_5, in_data_5 + bias_size);
    
    // Read batchnorm running_var
    std::ifstream ifile6("input6.data");
    if (!ifile6.is_open()) {
      std::cerr << "Error: Could not open input6.data file.\n";
      return 1;
    }
    float in_data_6[bias_size];
    for (unsigned i = 0; i < bias_size; i++) {
      ifile6 >> in_data_6[i];
    }
    size_t size_bytes_in6 = sizeof(float) * bias_size;
    std::vector<float, aligned_allocator<float>> source_in6(in_data_6, in_data_6 + bias_size);
    
    // Allocate output buffer with zeros
    size_t size_bytes_out = sizeof(float) * output_size;
    std::vector<float, aligned_allocator<float>> source_out(output_size, 0.0f);

    // OPENCL HOST CODE AREA START
    // get_xil_devices() is a utility API which will find the xilinx
    // platforms and will return list of devices connected to Xilinx platform
    auto devices = xcl::get_xil_devices();
    // read_binary_file() is a utility API which will load the binaryFile
    // and will return the pointer to file buffer.
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
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
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
    
    // Allocate Buffer in Global Memory
    // Buffers are allocated using CL_MEM_USE_HOST_PTR for efficient memory and
    // Device-to-host communication
    OCL_CHECK(err, cl::Buffer buffer_in0(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, size_bytes_in0, source_in0.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in1(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, size_bytes_in1, source_in1.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in2(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, size_bytes_in2, source_in2.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in3(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, size_bytes_in3, source_in3.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in4(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, size_bytes_in4, source_in4.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in5(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, size_bytes_in5, source_in5.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_in6(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, size_bytes_in6, source_in6.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_out(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, size_bytes_out, source_out.data(), &err));

    // Set kernel arguments
    OCL_CHECK(err, err = krnl_top.setArg(0, buffer_in0));  // input tensor
    OCL_CHECK(err, err = krnl_top.setArg(1, buffer_in1));  // conv weights
    OCL_CHECK(err, err = krnl_top.setArg(2, buffer_in2));  // conv bias
    OCL_CHECK(err, err = krnl_top.setArg(3, buffer_in3));  // batchnorm gamma
    OCL_CHECK(err, err = krnl_top.setArg(4, buffer_in4));  // batchnorm beta
    OCL_CHECK(err, err = krnl_top.setArg(5, buffer_in5));  // batchnorm running_mean
    OCL_CHECK(err, err = krnl_top.setArg(6, buffer_in6));  // batchnorm running_var
    OCL_CHECK(err, err = krnl_top.setArg(7, buffer_out));  // output tensor
    
    // Copy input data to device global memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_in0, buffer_in1, buffer_in2, 
                                                    buffer_in3, buffer_in4, buffer_in5, 
                                                    buffer_in6, buffer_out}, 0 /* 0 means from host*/));

    cl::Event event;
    uint64_t nstimestart, nstimeend;
    std::cout << "|-------------------------+-------------------------|\n"
              << "| Kernel                  |    Wall-Clock Time (ns) |\n"
              << "|-------------------------+-------------------------|\n";

    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_top, nullptr, &event));

    // Copy Result from Device Global Memory to Host Local Memory
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_out}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    // OpenCL Host Code Ends

    // Get the execution time
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_START, &nstimestart));
    OCL_CHECK(err, err = event.getProfilingInfo<uint64_t>(CL_PROFILING_COMMAND_END, &nstimeend));
    auto exe_time = nstimeend - nstimestart;

    std::cout << "| " << std::left << std::setw(24) << "top "
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
        ofile << source_out[i] << std::endl;
    }
    ofile.close();
    
    std::cout << "Output size: " << output_size << " elements written to output.data" << std::endl;
    return EXIT_SUCCESS;
}