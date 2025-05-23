# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#=============================================================================
# run.tcl 
#=============================================================================
# Project name
set hls_prj out.prj

# Open/reset the project
open_project ${hls_prj} -reset

open_solution -reset solution1 -flow_target vivado

# Top function of the design is "top"
set_top top

# Add design and testbench files
add_files kernel.cpp
add_files -tb host.cpp -cflags "-std=gnu++0x"
open_solution "solution1"

# Target device is u280
set_part {xcu280-fsvh2892-2L-e}

# Target frequency
create_clock -period 3.33

# Run HLS
csim_design -O

exit
