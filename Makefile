#Default log level is 5, which prints up to MESSAGE, 0 will only print critical errors and 14 will print everything up to the most low level debug information
LOG_LEVEL=5
#Uncomment to compile in double precision mode
#DOUBLE_PRECISION=-DDOUBLE_PRECISION
NVCC=nvcc
CXX=g++
UAMMD_ROOT=uammd
INCLUDEFLAGS=-I$(UAMMD_ROOT)/src -I$(UAMMD_ROOT)/src/third_party
NVCCFLAGS=-ccbin=$(CXX) -std=c++14 -O3 $(INCLUDEFLAGS) -DMAXLOGLEVEL=$(LOG_LEVEL) $(DOUBLE_PRECISION)

all: $(patsubst %.cu, %, $(wildcard *.cu))

%: %.cu Makefile
	$(NVCC) $(NVCCFLAGS) $< -o $(@:.out=) -lcufft

clean: $(patsubst %.cu, %.clean, $(wildcard *.cu))

%.clean:
	rm -f $(@:.clean=)
