# Compiler and flagg
CXX 		:=g++-14
CXXFLAGS 	:=-std=c++2c -Wall -O2 -pedantic #
EIGEN_PATH 	:=../../../CommonMathLib/eigen-3.4.0#
CXXFLAGS 	+= -I$(EIGEN_PATH)
LDFLAGS =#

# Directories
SRC_DIR = src
BUILD_DIR = build

# Find all source files (excluding main.cc)
SRC_FILES = $(filter-out $(SRC_DIR)/main.cc, $(wildcard $(SRC_DIR)/*.cc))
# Generate object file names
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cc,$(BUILD_DIR)/%.o,$(SRC_FILES))
# Main object file
MAIN_OBJ = $(BUILD_DIR)/main.o
# Final executable
TARGET = $(BUILD_DIR)/main.elf

# Default target
all: $(BUILD_DIR) $(TARGET)

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Rule to build the executable
$(TARGET): $(OBJ_FILES) $(MAIN_OBJ)
	$(CXX) $(LDFLAGS) -o $@ $^

# Rule to compile source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean target
clean:
	rm $(TARGET) $(OBJ_FILES) 

# Phony targets
.PHONY: all clean
