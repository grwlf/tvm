PYTHON ?= python

TVM_ROOT = $(shell cd ../../../; pwd)
NNVM_ROOT = $(TVM_ROOT)/nnvm
DMLC_ROOT = $(TVM_ROOT)/dmlc-core

X86_CFLAGS = \
	-std=c++11 -O0 -g -fPIC \
	-I$(TVM_ROOT) \
	-I$(NNVM_ROOT)/include \
	-I$(TVM_ROOT)/include \
	-I$(TVM_ROOT)/dlpack/include \
	-I$(DMLC_ROOT)/include\
	-DDMLC_LOG_STACK_TRACE=0 \

X86_LDFLAGS = \
	-L$(TVM_ROOT)/build

.PHONY: clean all

all: bin/run_model

lib/build_model.stamp: build_model.py
	@mkdir -p $(@D)
	$(PYTHON) build_model.py

.INTERMEDIATE: lib/build_model.stamp
lib/deploy_graph.json lib/deploy_params.bin lib/deploy_lib.o: lib/build_model.stamp

bin/run_model: main.cpp lib/deploy_graph.json lib/deploy_params.bin lib/deploy_lib.o
	@mkdir -p $(@D)
	$(CXX) $(X86_CFLAGS) $(X86_LDFLAGS) main.cpp lib/deploy_lib.o -o $@ -ltvm_runtime -lpthread -ldl \
		-Wl,--format=binary -Wl,lib/deploy_graph.json -Wl,lib/deploy_params.bin -Wl,--format=default

clean:
	rm -rf lib bin

