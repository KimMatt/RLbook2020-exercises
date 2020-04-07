# game.py
# Launches an interface to play against a tic tac toe agent

import pygame
import pickle
from pygame.locals import ( 
    QUIT,
    MOUSEBUTTONUP
)
from src.tictactoe import TicTacToe
from src.logger import Logger
from src.agent import Agent

def load_agent(filename):
    agent_policy = pickle.load(open("policies/" + filename + ".p", "rb"))
    agent = Agent(2, 0.0)
    agent.set_policy(agent_policy)
    return agent


if __name__ == "__main__":
    pygame.init()

    display_info = pygame.display.Info()
    DISPLAY_WIDTH = display_info.current_w
    DISPLAY_HEIGHT = display_info.current_h

    WINDOW_HEIGHT = int(DISPLAY_HEIGHT * (9/10))
    WINDOW_WIDTH = int(WINDOW_HEIGHT * (3/4))
    PADDING = int(WINDOW_WIDTH * 0.05)
    BOX_WIDTH = int((WINDOW_WIDTH - (PADDING * 2)) * (1/3))
    LINE_WIDTH = 3
    MOVE_WIDTH = 9
    WHITE = (255,255,255)

    def draw_clear_window(screen):
        screen.fill((0,0,0)) # fill screen with black        
        pygame.draw.line(screen, WHITE, (PADDING, PADDING + BOX_WIDTH), (PADDING + (BOX_WIDTH * 3), PADDING + BOX_WIDTH), LINE_WIDTH)
        pygame.draw.line(screen, WHITE, (PADDING, PADDING + BOX_WIDTH*2), (PADDING + (BOX_WIDTH * 3), PADDING + BOX_WIDTH*2), LINE_WIDTH)
        pygame.draw.line(screen, WHITE, (PADDING + BOX_WIDTH, PADDING), (PADDING + BOX_WIDTH, PADDING + BOX_WIDTH *3), LINE_WIDTH)
        pygame.draw.line(screen, WHITE, (PADDING + BOX_WIDTH*2, PADDING), (PADDING + BOX_WIDTH*2, PADDING + BOX_WIDTH *3), LINE_WIDTH)


    def get_pos(move):
        row = int(move/3)
        col = move%3
        pos = (PADDING + (col * BOX_WIDTH), PADDING + (row * BOX_WIDTH))
        return pos


    def get_box(pos):
        # return which box is clicked on
        if pos[0] > WINDOW_WIDTH - PADDING or pos[0] < PADDING or pos[1] < PADDING or pos[1] > (WINDOW_WIDTH + PADDING):
            return None
        else:
            pos_x = pos[0] - PADDING
            pos_y = pos[1] - PADDING
            column = int(pos_x / BOX_WIDTH) + 1
            row = int(pos_y / BOX_WIDTH) + 1
            return (column * row) + ((row-1) * (3-column)) - 1


    def draw_move(pos, player_n):
        pos_x = pos[0] - ((pos[0] - PADDING)%BOX_WIDTH)
        pos_y = pos[1] - ((pos[1] - PADDING)%BOX_WIDTH)
        box_padding = int(BOX_WIDTH * 0.1)
        if player_n == 1:
            pygame.draw.line(screen, WHITE, (pos_x+box_padding, pos_y+box_padding), 
                             (pos_x + BOX_WIDTH - box_padding, pos_y + BOX_WIDTH - box_padding),
                             MOVE_WIDTH)
            pygame.draw.line(screen, WHITE, (pos_x + BOX_WIDTH - box_padding, pos_y+box_padding), 
                             (pos_x + box_padding, pos_y + BOX_WIDTH - box_padding),
                             MOVE_WIDTH)
        else:
            pygame.draw.circle(screen, WHITE, (pos_x + int(BOX_WIDTH/2),pos_y + int(BOX_WIDTH/2)), 
                               int(BOX_WIDTH/2) - box_padding, MOVE_WIDTH)


    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    draw_clear_window(screen)

    agent = load_agent("meta_agent")
    logger = Logger()
    #TODO: Make logger optional
    game_model = TicTacToe(logger)
    agent.enter(game_model)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                box = get_box(pos)
                if box is not None and game_model.in_progress:
                    game_model.play_move(1,box)
                    print("Player made move {}",format(box))
                    draw_move(pos, 1)
                    if game_model.in_progress:
                        agent.play()
                        last_move = agent.get_last_move()
                        print("Agent made move {}",format(last_move))
                        print(game_model.game_state)
                        draw_move(get_pos(last_move), 2)

        pygame.display.flip() # flip everything to the display

