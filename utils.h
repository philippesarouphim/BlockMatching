inline bool checkBlockInBounds(int block_x, int block_y, int block_size, int width, int height){
    return block_x >= 0 &&
           block_x + block_size < width &&
           block_y >= 0 &&
           block_y + block_size < height;
}
