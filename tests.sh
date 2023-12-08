echo make
make
echo
echo ============================================

for blocks in 1 2 4 8 16 32 64 128 256 512 1024
do
    for threads in 1 2 4 8 16 32 64 128 256 512 1024
    do
        echo -n "(Blocks: $blocks, Threads: $threads): "
        ./out/tss ./data/mario.mp4 ./out/data/mario.mp4 8 4 $blocks $threads 10
    done
done