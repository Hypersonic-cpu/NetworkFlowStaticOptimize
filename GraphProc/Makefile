# Compiler and flags
CXX = g++-14
CXXFLAGS = -std=c++23 -Wall -Wextra -Werror -pedantic -g
LDFLAGS = #

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
	rm -rf $(BUILD_DIR)

# Phony targets
.PHONY: all clean
