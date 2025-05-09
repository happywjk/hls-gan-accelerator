


#=============================================================================
# run.tcl 
#=============================================================================
# Project name
set hls_prj gen_block.prj

# Open/reset the project
open_project ${hls_prj} -reset

open_solution -reset solution1 -flow_target vivado

# Top function of the design is "top"
set_top top

# Add design and testbench files
add_files gan_block.cpp

open_solution "solution1"

# Target device is u280
set_part {xcu280-fsvh2892-2L-e}

# Target frequency
create_clock -period 3.33
config_compile -pipeline_loops 0
# Run synthesis with top-level function 'top'
csynth_design
exit
