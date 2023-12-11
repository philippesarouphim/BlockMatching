CXX = nvcc
CXXFLAGS = 
SRC_DIR = .
OBJ_DIR = out/obj
BIN_DIR = out/
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
DATA_OUT_DIR = out/data/

# Default target
all: directories tss ds

# Rule to make the object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ -I/usr/include/opencv4 `pkg-config --libs opencv4`

# Rule to make the program
tss: $(OBJECTS)
	$(CXX) $(CXXFLAGS) three_step_search.cu $^ -o $(BIN_DIR)/$@ -I/usr/include/opencv4 `pkg-config --libs opencv4`

ds: $(OBJECTS)
	$(CXX) $(CXXFLAGS) diamond_search.cu $^ -o $(BIN_DIR)/$@ -I/usr/include/opencv4 `pkg-config --libs opencv4`

# Rule to make the necessary directories
directories:
	mkdir -p $(OBJ_DIR) $(BIN_DIR) $(DATA_OUT_DIR)

# Rule to clean the build files
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR) $(DATA_OUT_DIR)

.PHONY: all directories clean