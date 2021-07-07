# Competition_1v1snakes

### Baseline agents: 
1. random-agent 
2. greedy-agent
3. dqn-agent

### Start to train rl-agent

python agent/dqn/main.py

note: You can edit different parameters, for example

python agent/dqn/main.py --lr_a 0.001 --seed_nn 2

### Start to evaluate 

python evaluation.py --my_ai "random" --opponent "dqn"
