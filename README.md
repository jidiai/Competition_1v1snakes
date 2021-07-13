# Competition_1v1snakes

### Dependency
You need to create competition environment.
>conda create -n snake1v1 python=3.6

>conda activate snake1v1

>pip install -r requirements.txt

### Baseline agents
1. random-agent 
2. greedy-agent
3. rl-agent

### How to train rl-agent

>python rl_trainer/main.py

You can edit different parameters, for example

>python rl_trainer/main.py --lr_a 0.001 --seed_nn 2

### Evaluation 
You can locally evaluate your different agents

>python evaluation.py --my_ai "random" --opponent "dqn"

### How to test submission 
You can locally test your submission. At Jidi platform, we evaluate your submission as same as **run_log.py**

Once you run this file, you can locally check battle logs in the folder named "logs".

For example, 
>python run_log.py --my_ai "random" --opponent "dqn"

### Ready to submit 

1. Random policy --> **agent/random/submission.py**
2. Greedy policy --> **agent/greedy/submission.py**
3. RL policy --> **agent/dqn/submission.py** and **critic_5000.pth**

