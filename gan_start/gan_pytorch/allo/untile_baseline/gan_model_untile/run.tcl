# Define the project name
set hls_prj "gan_model_untile.prj"

# Open/reset the project
open_project ${hls_prj} -reset

open_solution -reset solution1 -flow_target vivado

# Top function of the design is "top"
set_top top

# Add design files one by one
add_files gan_model.cpp
add_files gan_block_0.cpp
add_files gan_block_1.cpp
add_files gan_block_2.cpp
add_files gan_block_3.cpp
add_files gan_block_4.cpp
add_files gan_block_5.cpp
add_files conv_transpose_untile.cpp
add_files batchnorm_untile.cpp
add_files relu.cpp
add_files utils.cpp
add_files tanh.cpp

open_solution "solution1"

# Target device is u280
set_part {xcu280-fsvh2892-2L-e}

# Target frequency
create_clock -period 3.33
config_compile -pipeline_loops 0
# Run synthesis with top-level function 'top'
csynth_design
exit