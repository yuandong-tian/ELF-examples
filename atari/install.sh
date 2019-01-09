#!/bin/bash

ROOT=`pwd`

git submodule init
git submodule update --recursive

cd $ROOT/../ELF
git submodule init
git submodule update --recursive
make elf

cd $ROOT/Arcade-Learning-Environment
mkdir build; cd build
cmake -DUSE_SDL=OFF -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=OFF ..
make -j 4

cd $ROOT
mkdir build; cd build
cmake ..
make -j4

for game in breakout montezuma_revenge seaquest; do
    wget -O ${game}.bin https://github.com/npow/atari/blob/master/roms/${game}.bin?raw=true
done
