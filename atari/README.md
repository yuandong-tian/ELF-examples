Atari
==============

Run the following command to compile:
```
sh ./install.sh
```

Then start training:
```
cd build
python -u ../run.py --gpu 0 --num_game_thread 128 --batchsize 32 --reward_clip 1
```
