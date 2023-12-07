CXX = nvcc
CXXFLAGS = -g
SRC_DIR = .
OBJ_DIR = out/obj
BIN_DIR = out/
SOURCES = $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/*.cu)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
DATA_OUT_DIR = out/data/

# Default target
all: directories three_step_search

# Rule to make the object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ -I/usr/include/opencv4 `pkg-config --libs opencv4`

# Rule to make the program
three_step_search: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $(BIN_DIR)/$@ -I/usr/include/opencv4 `pkg-config --libs opencv4`

# Rule to make the necessary directories
directories:
	mkdir -p $(OBJ_DIR) $(BIN_DIR) $(DATA_OUT_DIR)

# Rule to clean the build files
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(DATA_OUT_DIR)

.PHONY: all directories clean