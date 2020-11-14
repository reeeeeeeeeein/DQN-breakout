from collections import deque
import os
import random
from tqdm import tqdm
import math
import torch
import datetime
import pytz
from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory
import time

date=datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H_%M_%S")

GAMMA = 0.99
GLOBAL_SEED = 0
MEM_SIZE = 100_00 #
RENDER = False
PREFIX="./"+date
SAVE_PREFIX = PREFIX+"/models"
CONFIG_PATH=PREFIX+"/config.txt"
os.mkdir(PREFIX)
os.mkdir(SAVE_PREFIX)

STACK_SIZE = 4

EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 1000000

BATCH_SIZE = 16 #
POLICY_UPDATE = 8 #
TARGET_UPDATE = 10_000 #
WARM_STEPS = 0 #
MAX_STEPS = 5000000
EVALUATE_FREQ = 100_00

EPSILON = 0.01 #
ALPHA = 0.8 #
BETA = 0.4 #
BETA_INC = 0.001 #
with open(CONFIG_PATH,"a") as f:
    f.write("MEM_SIZE: "+str(MEM_SIZE)+"\n")
    f.write("ALPHA: "+str(ALPHA)+"\n")
    f.write("BETA: "+str(BETA)+"\n")
    f.write("BETA_INC: "+str(BETA_INC)+"\n")
    f.write("WARM_STEPS: "+str(WARM_STEPS)+"\n")
    f.write("POLICY_UPDATE: "+str(POLICY_UPDATE)+"\n")
    f.write("BATCH_SIZE: "+str(BATCH_SIZE)+"\n")
    f.write("TARGET_UPDATE: "+str(TARGET_UPDATE)+'\n')
rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)


torch.manual_seed(new_seed())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = MyEnv(device)
agent = Agent(
    env.get_action_dim(),
    device,
    GAMMA,
    new_seed(),
    EPS_START,
    EPS_END,
    EPS_DECAY,
)
memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device,EPSILON=EPSILON,ALPHA=ALPHA,BETA_INC=BETA_INC,BETA=BETA)

#### Training ####
obs_queue: deque = deque(maxlen=5)
done = True

progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                   ncols=50, leave=False, unit="b")
for step in progressive:
    if done:
        observations, _, _ = env.reset()
        for obs in observations:
            obs_queue.append(obs)

    training = len(memory) > WARM_STEPS
    state = env.make_state(obs_queue).to(device).float()
    action = agent.run(state, training)
    obs, reward, done = env.step(action)
    obs_queue.append(obs)
    memory.push(env.make_folded_state(obs_queue), action, reward, done, agent)

    if step % POLICY_UPDATE == 0 and training:
        agent.learn(memory, BATCH_SIZE)

    if step % TARGET_UPDATE == 0:
        agent.sync()

    if step % EVALUATE_FREQ == 0:
        avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
        with open(PREFIX+"/rewards.txt", "a") as fp:
            fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
        if RENDER:
            prefix = f"eval_{step//EVALUATE_FREQ:03d}"
            os.mkdir(prefix)
            for ind, frame in enumerate(frames):
                with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                    frame.save(fp, format="png")
        agent.save(os.path.join(
            SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
        done = True
