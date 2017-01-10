open_project sort_syn

add_files sort.c

set_top workload

open_solution -reset solution
set_part virtex7
create_clock -period 10
#csim_design
csynth_design
#cosim_design -rtl verilog -tool modelsim -trace_level all

exit
