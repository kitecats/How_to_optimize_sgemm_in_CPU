TARGET=test_bl_sgemm_step4

#CROSS_COMPILE ?= riscv64-unknown-linux-gnu-
CROSS_COMPILE ?=

CC  = $(CROSS_COMPILE)gcc
CXX = $(CROSS_COMPILE)g++

COMPILER_OPT_LEVEL=O3

ifeq ($(CROSS_COMPILE),)
CFLAGS = -$(COMPILER_OPT_LEVEL) -march=core-avx2 -fPIC  -fopenmp
LDLIBS = -lpthread -lm
else
CFLAGS = -$(COMPILER_OPT_LEVEL) -march=rv64imafdcv -mabi=lp64d  -fopenmp
LDLIBS = -lm
endif

COMMONDIR = ../common
CFLAGS += -I$(COMMONDIR)

ASM_SRC= 
CC_SRC= my_sgemm.c \
        $(COMMONDIR)/test_bl_sgemm.c \
        $(COMMONDIR)/bl_sgemm_ref.c \
        $(COMMONDIR)/bl_sgemm_util.c

CPP_SRC=

ALL_ASM_OBJS := $(ASM_SRC:=.o)
ALL_C_OBJS := $(CC_SRC:=.o)
ALL_CXX_OBJS := $(CPP_SRC:=.o)

MKDEP_OPT = -MMD -MF $@.d

ALL_OBJS += $(ALL_ASM_OBJS) $(ALL_C_OBJS) $(ALL_CXX_OBJS)
ALL_DEPS := $(ALL_OBJS:=.d)

BLISLAB_TEST_EXE=$(TARGET:=.x)

.PHONY: all clean

all: $(BLISLAB_TEST_EXE)

# include dependency files of application
ifneq ($(MAKECMDGOALS),clean)
-include $(ALL_DEPS)
endif

$(BLISLAB_TEST_EXE): $(ALL_OBJS)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

# ---------------------------------------------------------------------------
# Object files compiling rules
# ---------------------------------------------------------------------------
%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS) $(MKDEP_OPT)

%.cpp.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@ $(LDFLAGS) $(MKDEP_OPT)
# ---------------------------------------------------------------------------

clean:
	@rm -rf $(ALL_OBJS) $(SHAREDLIBBLISLAB) $(BLISLAB_TEST_EXE) $(ALL_DEPS)


