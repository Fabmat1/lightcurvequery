UNAME := $(shell uname -s)

CXX      := g++
CXXFLAGS := -O3 -march=x86-64-v2 -fPIC -shared -fopenmp -std=c++17
EIGEN    := $(shell pkg-config --cflags eigen3 2>/dev/null || echo "-I/usr/include/eigen3")
LDFLAGS  := -lfftw3

SRC      := gls_fast.cpp
TARGET_BASE := libgls_fast

ifeq ($(UNAME),Darwin)
    CXX      := g++-14
    CXXFLAGS := -O3 -fPIC -shared -std=c++17 -Xpreprocessor -fopenmp
    LDFLAGS  += -lomp
    TARGET   := $(TARGET_BASE).dylib
else ifeq ($(OS),Windows_NT)
    TARGET  := $(TARGET_BASE).dll
else
    TARGET  := $(TARGET_BASE).so
endif

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(EIGEN) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET_BASE).so $(TARGET_BASE).dylib $(TARGET_BASE).dll