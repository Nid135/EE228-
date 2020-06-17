import numpy as np
import h5py
from keras.models import load_model
import time
import csv

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction

class MyAgent(Agent):

    def __init__(self, game, display=None):
        self.mymodel2 = load_model('model2.h5')
        super().__init__(game, display)
        
    def step(self):
        start=time.time()
        board=self.game.board
        size=(4,4)
        d={}
        for i in range(1,16):
            d[2**i]=i
        d[0]=00
        onehot=np.zeros(shape=size + (16,), dtype=bool)
        for i in range(size[0]):
            for j in range(size[1]):
                onehot[i,j,d[board[i,j]]]=1
        pre=self.mymodel2.predict(np.array([onehot]))[0]
        direction=np.argmax(pre)
        steptime=time.time()-start
        f=open('time.csv','a')
        writer=csv.writer(f)
        writer.writerow([steptime])
        f.close()
        print("time of one step:",steptime)
        return direction
        
