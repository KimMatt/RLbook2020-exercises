# draw_policies.py
#
# draw the path of a greedy policy based on the epsilon greedy value map

import pygame
import pickle
import numpy as np
from pygame.locals import (
    QUIT,
    MOUSEBUTTONUP
)
from racetrack import RaceTrack, Learner, Agent

BOX_SIZE = 15
WINDOW_WIDTH = 500
WINDOW_HEIGHT = 500
BLACK = (0,0,0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
LINE_WIDTH = 2

VALUES = pickle.load(open("./pickles/race_values.p", "rb"))

def draw_optimal_path(start_pos, values, track, screen):
    learner = Learner()
    agent = Agent(track)
    agent.position = start_pos
    end_of_path = False
    position = start_pos
    while not end_of_path:
        max_action = None
        max_action_value = values[str((np.array(position),np.array(learner.actions[0])))]
        result_position = None
        for action in learner.actions:
            print(action)
            print(values[str((position, action))])
            if values[str((position, action))] >= max_action_value:
                max_action_value = values[str((position, action))]
                max_action = action
        agent.play(max_action)
        result_position = agent.position
        print("drawing line from {} to {}".format(position,result_position))
        pygame.draw.line(screen, RED, (position[0] * BOX_SIZE + (BOX_SIZE/2), position[1] * BOX_SIZE + (
            BOX_SIZE/2)), (result_position[0] * BOX_SIZE + (BOX_SIZE/2), result_position[1] * BOX_SIZE + (BOX_SIZE/2)))
        result = agent.play(max_action)
        if result == 1:
            end_of_path = True
        position = result_position

def draw_box(screen,x,y):
    pygame.draw.line(screen, BLACK, (x*BOX_SIZE,y*BOX_SIZE), (x*BOX_SIZE, y*BOX_SIZE + BOX_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, BLACK, (x*BOX_SIZE, y*BOX_SIZE), (x*BOX_SIZE + BOX_SIZE, y*BOX_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, BLACK, (x*BOX_SIZE + BOX_SIZE, y*BOX_SIZE), (x*BOX_SIZE + BOX_SIZE, y*BOX_SIZE + BOX_SIZE), LINE_WIDTH)
    pygame.draw.line(screen, BLACK, (x*BOX_SIZE, y*BOX_SIZE + BOX_SIZE), (x*BOX_SIZE + BOX_SIZE, y*BOX_SIZE + BOX_SIZE), LINE_WIDTH)

if __name__ == "__main__":
    pygame.init()

    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    SCREEN.fill(WHITE)
    pygame.display.flip()

    MAP = open("./map.bin", "r").readlines()
    for y, line in enumerate(MAP):
        for x, sign in enumerate(line):
            if sign == str(1):
                draw_box(SCREEN, x, y)
                pygame.display.flip()

    track = RaceTrack()
    #for start_pos in track.start_positions.values():
    draw_optimal_path(np.array([7,31]), VALUES, track, SCREEN)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
        pygame.display.flip()
