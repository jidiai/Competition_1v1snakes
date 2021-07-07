# Competition_1v1snakes

### Baseline agents: 
1. random-agent 
2. greedy-agent
3. ddpg-agent

### Start to train ddpg-agent

python agent/ddpg/main.py

note: You can edit different arguments, for example
python agent/ddpg/main.py --lr_a 0.001 --seed_nn 2

### Start to evaluate 

python evaluation.py --my_ai "random" --opponent "ddpg"
