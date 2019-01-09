Atari
==============

Run the following command to compile:
```
sh ./install.sh
```

Then start training:
```
cd build
python -u ../run.py --gpu 0 --num_game_thread 1024 --batchsize 128 --reward_clip 1 --adv_clip 1.0 --ratio_clamp 1.5
```
