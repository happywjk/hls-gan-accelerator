def Convolution_layer_unoptimized():
    # concrete_type, batch_size, channel_in, channel_out, height, width,kernel_height, kernel_width,height_out,widthout,stride = int32,16,3,128,258,258,4,4,128,128,2
    concrete_type, batch_size, channel_in, channel_out, height, width,height_after_padding, width_after_padding,kernel_height, kernel_width,height_out,widthout,stride,padding = float32,4,3,16,256,256,258,258,4,4,128,128,2,1
    kernel_out =1
    # def add_padding(x,padding):
    #     return np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)
    def kernel_convolution_tile[Ty, batch, cin, cout, height,width,heigth_after_padding, width_after_padding,kernel_height,kernel_width,hout,wout,](
            input: "Ty[ cin, kernel_height, kernel_width]", Weight: "Ty[cout,cin, kernel_height, kernel_width]", Bias: "Ty[cout]",out:"Ty[cout,kernel_out,kernel_out]"
            ):
        padded_input:Ty[batch, cin,heigth_after_padding,width_after_padding] = 0
        for i, m, n, l in dsl.grid(batch, cin, height, width):
            padded_input[i, m, n + padding, l + padding] = input[i, m, n, l]

        for j in dsl.grid(cout):
                out[j,0,0] = Bias[j]  # Initialize with bias                
                for m in range(cin):  # Iterate over input channels
                    for p in range(kernel_height):
                        for q in range(kernel_width,name="inner_loop"):
                                out[ j,0,0] += Weight[j, m, p, q] * padded_input[m, p, q]
    
    s_linear = allo.customize(kernel_convolution_layer, instantiate=[concrete_type, batch_size, channel_in, channel_out, height, width,height_after_padding, width_after_padding,kernel_height, kernel_width,height_out,widthout])

    def top[Ty, batch, cin, cout, height,width,height_after_padding, width_after_padding,kernel_height,kernel_width,hout,wout](
            input: "Ty[batch, cin, height, width]", Weight: "Ty[cout, cin,kernel_height, kernel_width]", Bias: "Ty[cout]", out:"Ty[batch, cout,hout,wout]"
    ):
        for i,k,z in dsl.grid(batch,height_after_padding,width_after_padding):
             for m,n,l in dsl.grid(cin,kernel_height,kernel_width)
                

    s = allo.customize(top, instantiate=[concrete_type,  batch_size, channel_in, channel_out, height,width,height_after_padding, width_after_padding,kernel_height,kernel_width,height_out,widthout])
    s.compose(s_linear)

    return s.build(target="vitis_hls", mode="csyn", project="convolution_unopitimized_new.prj")

def test_convolution_layer():
    # Parameters for small test size
    # concrete_type, batch_size, channel_in, channel_out, height, width,kernel_height, kernel_width,height_out,widthout,stride,padding = int32,16,3,128,258,258,4,4,128,128,2,1
    concrete_type, batch_size, channel_in, channel_out, height, width,height_after_padding, width_after_padding,kernel_height, kernel_width,height_out,widthout,stride,padding = float32,4,3,16,256,256,258,258,4,4,128,128,2,1
    total_time = 0
    # Random input, weight, and bias initialization
    X = np.random.randn(batch_size, channel_in,height,width).astype(np.float32)
    W = np.random.randn(channel_out, channel_in,kernel_height,kernel_width).astype(np.float32)
    B = np.random.randn(channel_out).astype(np.float32)
    allo_C = np.zeros((batch_size, channel_out,height_out,widthout), dtype=np.float32)
     
    # Instantiate the layer using systolic optimizations
    mod = Convolution_layer_unoptimized()
    print("finish compilation")
    print("Finish compilation, checking mod type:", type(mod))
    start_time = time.time()
    mod(X, W,B,allo_C)
    end_time = time.time()
    total_time += (end_time - start_time)
    print(f"Total time : {total_time:.5f} seconds")
    # ref = numpy_linear_layer(X, W, B)
    Convolution_layer = nn.Conv2d(3, 16,4,2,1)
    with torch.no_grad():  # Avoid tracking gradients during this operation
        Convolution_layer.weight.copy_(torch.from_numpy(W))
        Convolution_layer.bias.copy_(torch.from_numpy(B))
    
    ref = Convolution_layer(torch.from_numpy(X)).detach().numpy()
    # Verify with NumPy reference
    # Verify with NumPy reference
    print("start comparing")
    np.testing.assert_allclose(allo_C, ref, rtol=1e-05,atol=1e-3)
    print("Test Passed!")


test_convolution_layer()