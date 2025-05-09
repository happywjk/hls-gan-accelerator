#!/bin/bash
echo "Setting up Vitis 2022.1 located under /opt/xilinx..."
unset LM_LICENSE_FILE
#floating license: departemental server
export XILINXD_LICENSE_FILE=2100@flex.ece.cornell.edu
#Setup path
export VITIS=/opt/xilinx/2022.1/Vitis/2022.1
source $VITIS/settings64.sh
echo "Setting up Xilinx xrt for programming Xilinx boards ..."
source /opt/xilinx/xrt/setup.sh
export HLS_INCLUDE=/opt/xilinx/2022.1/Vitis_HLS/2022.1/include
export LD_LIBRARY_PATH=/opt/xilinx/2022.1/Vitis/2022.1/lib/lnx64.o:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/xilinx/2022.1/Vitis/2022.1/lib/lnx64.o/Default:$LD_LIBRARY_PATH
export XDEVICE="/opt/xilinx/platforms/xilinx_u250_gen3x16_xdma_4_1_202210_1/xilinx_u250_gen3x16_xdma_4_1_202210_1.xpfm"
# export XDEVICE="/opt/xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1/xilinx_u280_gen3x16_xdma_1_202211_1.xpfm"
source /opt/rh/devtoolset-7/enable
echo "Done!"