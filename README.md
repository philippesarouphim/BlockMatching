# Parallelization of Block-Matching Algorithm

All code is original except for the `gputimer.h` file provided in the previous labs.

### Requirements:
- CUDA-compatible GPU.
- Having the opencv4 C++ library installed on your computer (only used to read/write videos to the disk).

### Building
To build this project, a makefile was provided that builds both the `tss` (three_step_search) and the `ds` (diamond search) programs.

This makefile was developed for Linux systems. If you are trying to run it on another platform, you might run into issues. In this case you can simply compile all files using the `nvcc` compiler and you should be good to go.

### Usage
##### Three-step search program
`tss <video_in_path> <video_out_path> <block_size> <step_size> <num_blocks> <num_threads> <frames_limit>`
##### Diamond search program
`ds <video_in_path> <video_out_path> <block_size> <num_blocks> <num_threads> <frames_limit>`