Atari
==============

How to run
----------

Run the following command to compile:
```
sh ./install.sh
```

Then start training using A2C:
```
cd build
python -u ../run.py --gpu 0 --num_game_thread 1024 --batchsize 128 --reward_clip 1 --adv_clip 1.0 --ratio_clamp 1.5
```

You can add `--freq_update 1` to specify how frequent the actor is updated to match with the train models.

Performance
-----------

![Breakout Performance](imgs/breakout.png "")
