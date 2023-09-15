# multi-depot-cvrp
multi-depot-cvrp using Attention-PtrNet and Actor-Critic 

# Installation
python 3.10.13

```bash
pip install -r requirements.txt
```

# Training
```bash
python trainer.py
```
To see the training progress, run tensorboard (default logdir is logs)
```bash
tensorboard --logdir=logs
```

# Testing
```bash
python tester.py
```
