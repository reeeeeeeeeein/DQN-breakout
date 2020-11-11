from typing import (
    Tuple,
)

import torch

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)

import math
import numpy as np
from copy import deepcopy as copy


class treearray:
    def __init__(self, capacity):
        self.__capacity = capacity
        self.__bits = math.ceil(math.log(capacity, 2))
        self.__max_len = 1 << self.__bits
        self.__array = torch.zeros((self.__max_len + 1, 1), dtype=torch.float)

    def add(self, loc, val):
        # 单点加法
        while loc < self.__max_len:
            self.__array[loc] += val
            loc += loc & (-loc)

    def get_array(self):
        return self.__array

    def get_prefix_sum(self, loc):
        # 得到一个前loc个值的和
        val = 0
        while loc != 0:
            val += self.__array[loc]
            loc -= loc & (-loc)
        return val

    def change(self, loc, val):
        # 单点修改，不过要先查询之前的值才能加上去
        nowval = self.get_prefix_sum(loc) - self.get_prefix_sum(loc - 1)
        #print(val,nowval)
        self.add(loc, val - nowval)

    def search(self):
        # 进行采样
        sub_val = (1 << (self.__bits - 1))
        right = self.__max_len
        right_val = copy(self.__array[right])
        while sub_val != 0:
            left = right - sub_val
            left_val = copy(self.__array[left])
            if np.random.rand() < left_val / right_val:
                right = left
                right_val = left_val
            else:
                right_val -= left_val
            sub_val //= 2
        return right


class ReplayMemory(object):
    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.__size = 0
        self.__pos = 0

        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)
        self.__treearray = treearray(capacity)

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
            agent,
    ) -> None:
        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done
        indices = [self.__pos]  # torch.randint(0, high=self.__size, size=(batch_size,))
        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
        pri_val=agent.get_pri_val(b_state, b_action, b_reward, b_done, b_next)
        pri_val=pri_val.cpu().reshape((1))
        print(pri_val,b_reward)
        self.__treearray.change(self.__pos+1,pri_val)
        self.__pos = (self.__pos + 1) % self.__capacity
        self.__size = max(self.__size, self.__pos)

    def sample(self, batch_size: int) -> Tuple[
        BatchState,
        BatchAction,
        BatchReward,
        BatchNext,
        BatchDone,
    ]:
        indices = [self.__treearray.search()-1 for i in range(batch_size)]#torch.randint(0, high=self.__size, size=(batch_size,))
        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
        return b_state, b_action, b_reward, b_next, b_done

    def __len__(self) -> int:
        return self.__size
