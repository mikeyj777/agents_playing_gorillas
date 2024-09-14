import os
import math
import multiprocessing.pool
import random
from trajectory import trajectory_with_drag
import itertools
import numpy as np
import multiprocessing

from gorilla import Gorilla
from consts import Consts

class Gorillas_Game:
    '''
    
    gorilla game
    2-player game, but will train multiple gorillas
    reinforcement learning
    state:  distance, target_y, wind_speed.  one gorilla has the positive values associated with these.  
    the other is negative
    actions: angle (in degrees), and speed
    
    '''
    
    def __init__(self, num_players = 2):
        self.num_players = num_players
        self.players = []
        upwind = True
        for _ in range(self.num_players):
            self.players.append(Gorilla(upwind=upwind))
            upwind = not upwind

    def start_the_action(self, player):
        player.get_action()

    def initialize_game(self):
        self.wind_speed = random.randint(0, 10)
        player_1 = random.randint(0, self.num_players-1)
        player_2 = random.randint(0, self.num_players-1)
        while player_2 == player_1:
            player_2 = random.randint(0, self.num_players-1)
        self.g1:Gorilla = self.players[player_1]
        self.g2:Gorilla = self.players[player_2]
        self.distance = abs(self.g1.x_coord - self.g2.x_coord)
        self.g1.target_y = self.g2.height
        banana_to_target = math.sqrt(self.distance ** 2 + self.g1.target_y ** 2)
        self.g1.state = {
            'distance': self.distance,
            'wind_speed': self.wind_speed,
            'target_y': self.g1.target_y,
            'banana_to_target': banana_to_target
        }
        self.g2.target_y = self.g1.height
        self.g2.state = {
            'distance': -self.distance,
            'wind_speed': -self.wind_speed,
            'target_y': self.g2.target_y,
            'banana_to_target': banana_to_target
        }

    
    def play_game(self):
        keep_playing = True
        while keep_playing:
            self.initialize_game()
            self.players = [self.g1, self.g2]
            for player in self.players:
                player.get_action()
            # with multiprocessing.Pool() as p:
            #     pid = os.getpid()
            #     print(f'this is the process with id {pid}')
            #     _ = p.map(self.start_the_action, self.players)

if __name__ == '__main__':
    game = Gorillas_Game()
    game.play_game()
