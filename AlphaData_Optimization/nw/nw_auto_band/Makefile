PROJ=nw

all:
	sdaccel compile_bitstream.tcl
clean:
	rm -rf sdaccel* ${PROJ}/ .Xil/ output.data rose* *.o gmon.out
run:
	./${PROJ}/pkg/pcie/nw.exe input.data check.data ./${PROJ}/pkg/pcie/workload.xclbin
