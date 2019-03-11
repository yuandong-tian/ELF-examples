Simple Usage  
============

Compile with `mkdir build; cd build; cmake .. ; make`.

Then in `./build`, run:

```
python ../run.py --batchsize 4 --num_game_thread 1 --num_thread 8 --num_rollout_per_thread 100 --num_rollout_per_batch 1 --discount_factor 0.9
```
