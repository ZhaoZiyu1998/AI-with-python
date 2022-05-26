import torch
import random
import numpy as np
from game import SnakeGameAI, Direction, Point,BLOCK_SIZE
from collections import deque
from model import Linear_QNet,QTrainer
from helper import plot
MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.eps=0 ##randomness

        self.gam =0.9 ##discount factor
        self.memory = deque(maxlen=MAX_MEMORY) ##popleft
        self.model=Linear_QNet(11,256,3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gam)

    def get_state(self, game):
        head = game.snake[0]
        lpoint=Point(head.x-BLOCK_SIZE,head.y)
        rpoint=Point(head.x+BLOCK_SIZE,head.y)
        upoint=Point(head.x,head.y-BLOCK_SIZE)
        dpoint=Point(head.x,head.y+BLOCK_SIZE)
        ldir=game.direction==Direction.LEFT
        rdir=game.direction==Direction.RIGHT
        udir=game.direction==Direction.UP
        ddir=game.direction==Direction.DOWN
        state=[
            ##Danger straight
            (rdir and game.is_collision(rpoint)) or (ldir and game.is_collision(lpoint))
            or (udir and game.is_collision(upoint)) or (ddir and game.is_collision(dpoint)),
            ##Danger left

            (udir and game.is_collision(rpoint)) or (ddir and game.is_collision(lpoint))
            or (ldir and game.is_collision(upoint)) or (rdir and game.is_collision(dpoint)),
            ##Danger right
            (ddir and game.is_collision(rpoint)) or (udir and game.is_collision(lpoint))
            or (rdir and game.is_collision(upoint)) or (ldir and game.is_collision(dpoint)),

            ldir,
            rdir,
            udir,
            ddir,
            ## food location
            game.food.x<game.head.x,
            game.food.x>game.head.x,
            game.food.y<game.head.y,
            game.food.y>game.head.y,
            ]
        return np.array(state,dtype=int)



    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def train_long_memory(self):
        if len(self.memory)>BATCH_SIZE:
            mini_sample=random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample=self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)
    def get_action(self, state):
        ##random moves: tradeoff between randomness and exploitation
        self.eps=80-self.n_games
        final_move=[0,0,0]
        if random.randint(0,200)<self.eps:
            moverand=random.randint(0,2)
            final_move[moverand]=1
        else:
            state0=torch.tensor(state,dtype=torch.float)
            prediction=self.model(state0)
            move=torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

def train():
    plot_score = []
    plot_mean = []
    total=0
    record=0
    agent=Agent()
    game = SnakeGameAI()
    while True:
        ## get state
        prev_state = agent.get_state(game)
        ## get action
        last_move=agent.get_action(prev_state)
        ## next step
        reward,done,sroce=game.play_step(last_move)
        state_new=agent.get_state(game)
        ## short memory
        agent.train_short_memory(prev_state,last_move,reward,state_new,done)
        ## remember
        agent.remember(prev_state,last_move,reward,state_new,done)
        if done:
            ## train_long_memory and plot
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if sroce>record:
                record=sroce
                agent.model.save()
            print('Game: %s Score: %s Record: %s' % (agent.n_games, sroce, record))
            plot_score.append(sroce)
            total += sroce
            mean_score = total / agent.n_games
            plot_mean.append(mean_score)

            plot(plot_score, plot_mean)

if __name__ == '__main__':
    train()

