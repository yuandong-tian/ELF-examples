ELF RL package
==============

Run the script
```
./install.sh
```

Then add ELF python path
```
export PYTHONPATH=<ELF_DIR>/build/elf:$PYTHONPATH
export PYTHONPATH=<ELF_DIR>/src_py:$PYTHONPATH
```

Then try running it
```
cd build
python -u ../test_elf_python_old.py --gpu 0 --num_game_thread 128 --batchsize 32 --reward_clip 1
```

Note that this uses an old version of A2C (please `git pull` from ELF repo to get the most recent commit from `experimental branch`), which achieves decent performance after 8 hours of training. 
