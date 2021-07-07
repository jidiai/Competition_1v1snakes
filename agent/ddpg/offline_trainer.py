import os

cmd = 'nohup /home/mingchi/miniconda3/bin/python ' \
      '/home/mingchi/Project/Snake1v1/agent/ddpg/main.py > output.txt &'
os.system(cmd)
