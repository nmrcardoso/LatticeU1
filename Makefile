############################################################################################################
#VERSION  = V1_$(shell date "+%d_%m_%Y_%T")
VERSION  = V1_$(shell date "+%Y_%m_%d_%H-%M-%S")
STANDARD = c99
############################################################################################################

INC_DIR = 
INC_LIBS = 



GCC ?= g++
# -std=c++11


GPU_ARCH = sm_35
CUDA_PATH ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
GENCODE_FLAGS = -arch=$(GPU_ARCH)
LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart  -lcuda -lcurand
CCFLAGS   :=  -O3 -I$(CUDA_INC_PATH)/
NVCC      ?= $(CUDA_BIN_PATH)/nvcc


COMP_CAP = $(GPU_ARCH:sm_%=%0)
NVCCFLAGS := -m64 -O3   -Xcompiler -fopenmp  -I$(CUDA_INC_PATH)/
NVCCFLAGS += -D__COMPUTE_CAPABILITY__=$(COMP_CAP)
CCFLAGS += -D__COMPUTE_CAPABILITY__=$(COMP_CAP) 

SRCDIR=src
OBJDIR=obj
PROJECTNAME=u1
INCLUDES := -I. -I./include/ 
############################################################################################################
#For tune file
CUDA_VERSION = $(shell awk '/\#define CUDA_VERSION/{print $$3}' $(CUDA_PATH)/include/cuda.h)
HASH = \"cpu_arch=$(strip $(OS_ARCH)),gpu_arch=$(strip $(GPU_ARCH)),cuda_version=$(strip $(CUDA_VERSION))\"

all : directories $(PROJECTNAME)

OBJS := timer.o alloc.o parameters.o random.o update.o multilevel.o plaquette.o polyakov.o tune.o cuda_error_check.o actime.o plaquette_comps.o chromofieldPP.o chromofieldWL.o multilevel_TTO.o convertOrder.o smearing.o wilsonloop.o ../u1.o

INCS:= include
U1OBJS  := $(patsubst %.o,$(OBJDIR)/%.o,$(OBJS))
deps += $(U1OBJS:.o=.d)

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	@echo ""
	@echo "######################### Compiling: "$<" #########################"
	$(VERBOSE)mkdir -p $(dir $(abspath $@ ) )
	$(VERBOSE)$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) $(INCLUDES) -M $< -o ${@:.o=.d} -odir $(@D)
	$(VERBOSE)$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) $(INCLUDES) -o $@ -dc $<
#	$(VERBOSE)$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(EXTRA_NVCCFLAGS) $(INCLUDES) -dc $< -o $@
	
############################################################################################################CUDAOBJ=$
$(OBJDIR)/dlink.o: $(U1OBJS)
	@echo ""
	@echo ""######################### Creating: "dlink.o" #########################"
	$(VERBOSE)$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -dlink -o $@ $^ -lcudavert
	
$(OBJDIR)/tune.o: $(SRCDIR)/tune.cpp
	@echo ""
	@echo "######################### Compiling: "$<" #########################"
	$(VERBOSE)$(GCC) $(CCFLAGS) $(INC_DIR) $(INCLUDES) -DU1_HASH=$(HASH)  -MMD -MP   -c $< -o $@  
	
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@echo ""
	@echo "######################### Compiling: "$<" #########################"
	$(VERBOSE)$(GCC) $(CCFLAGS) $(INC_DIR) $(INCLUDES) -MMD -MP -c $< -o $@ 

$(PROJECTNAME):  $(U1OBJS) $(OBJDIR)/dlink.o
	@echo ""
	@echo "######################### Compiling: "$(PROJECTNAME)" #########################"
	$(VERBOSE)$(GCC) $(CCFLAGS)  -o $@ $+ $(INC_LIBS)  -fopenmp -L$(LIBDIR)/  $(LDFLAGS)

############################################################################################################
############################################################################################################
directories:
	$(VERBOSE)mkdir -p $(OBJDIR)
clean:
	$(VERBOSE)rm -r -f $(OBJDIR)
	$(VERBOSE)rm -f $(PROJECTNAME) $(MAINOBJ) $(OBJS) *.d

pack: 
	@echo Generating Package u1_$(VERSION).tar.gz
	@tar cvfz u1_$(VERSION).tar.gz *.cpp $(INCS)/*  $(SRCDIR)/* Makefile
	@echo Generated Package sun_$(VERSION).tar.gz

.PHONY : clean cleanall pack directories lib $(PROJECTNAME)

-include $(deps)
