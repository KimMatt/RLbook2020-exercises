# game.py
# Launches an interface to play against a tic tac toe agent

import os
import time
import pickle
import pygame
from pygame.locals import (
    QUIT,
    MOUSEBUTTONUP
)
from src.tictactoe import TicTacToe
from src.logger import Logger
from src.agent import Agent


def load_agent(filename):
    agent_policy = pickle.load(open("policies/" + filename + ".p", "rb"))
    agent = Agent(2, 0.0, True)
    agent.set_policy(agent_policy)
    return agent


if __name__ == "__main__":
    os.environ['SDL_VIDEO_CENTERED'] = '1'
    pygame.font.init()
    comic_sans = pygame.font.SysFont('Comic Sans MS', 30)
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
    RED = (255,0,0)
    GREEN = (0,255,0)

    logger = Logger()


    def draw_window(screen):
        pygame.draw.line(screen, WHITE, (PADDING, PADDING + BOX_WIDTH), (PADDING + (BOX_WIDTH * 3), PADDING + BOX_WIDTH), LINE_WIDTH)
        pygame.draw.line(screen, WHITE, (PADDING, PADDING + BOX_WIDTH*2), (PADDING + (BOX_WIDTH * 3), PADDING + BOX_WIDTH*2), LINE_WIDTH)
        pygame.draw.line(screen, WHITE, (PADDING + BOX_WIDTH, PADDING), (PADDING + BOX_WIDTH, PADDING + BOX_WIDTH *3), LINE_WIDTH)
        pygame.draw.line(screen, WHITE, (PADDING + BOX_WIDTH*2, PADDING), (PADDING + BOX_WIDTH*2, PADDING + BOX_WIDTH *3), LINE_WIDTH)
        win_surface = comic_sans.render('Wins: {}'.format(logger.agent_1_wins), False, WHITE)
        loss_surface = comic_sans.render('Loss: {}'.format(logger.agent_2_wins), False, WHITE)
        tie_surface = comic_sans.render('Ties: {}'.format(logger.ties), False, WHITE)
        screen.blit(win_surface, (PADDING, WINDOW_WIDTH + (PADDING*2)))
        screen.blit(loss_surface, (PADDING, WINDOW_WIDTH + (PADDING*3)))
        screen.blit(tie_surface, (PADDING, WINDOW_WIDTH + (PADDING*4)))


    def draw_colored_window(screen, color):
        pygame.draw.line(screen, color, (PADDING, PADDING + BOX_WIDTH), (PADDING + (BOX_WIDTH * 3), PADDING + BOX_WIDTH), LINE_WIDTH + 1)
        pygame.draw.line(screen, color, (PADDING, PADDING + BOX_WIDTH*2), (PADDING + (BOX_WIDTH * 3), PADDING + BOX_WIDTH*2), LINE_WIDTH + 1)
        pygame.draw.line(screen, color, (PADDING + BOX_WIDTH, PADDING), (PADDING + BOX_WIDTH, PADDING + BOX_WIDTH *3), LINE_WIDTH + 1)
        pygame.draw.line(screen, color, (PADDING + BOX_WIDTH*2, PADDING), (PADDING + BOX_WIDTH*2, PADDING + BOX_WIDTH *3), LINE_WIDTH + 1)


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


    def draw_move(pos, object_n, color=None):
        pos_x = pos[0] - ((pos[0] - PADDING)%BOX_WIDTH)
        pos_y = pos[1] - ((pos[1] - PADDING)%BOX_WIDTH)
        box_padding = int(BOX_WIDTH * 0.1)
        if object_n == 1:
            pygame.draw.line(screen, WHITE, (pos_x+box_padding, pos_y+box_padding),
                             (pos_x + BOX_WIDTH - box_padding, pos_y + BOX_WIDTH - box_padding),
                             MOVE_WIDTH)
            pygame.draw.line(screen, WHITE, (pos_x + BOX_WIDTH - box_padding, pos_y+box_padding),
                             (pos_x + box_padding, pos_y + BOX_WIDTH - box_padding),
                             MOVE_WIDTH)
        elif object_n == 2:
            pygame.draw.circle(screen, WHITE, (pos_x + int(BOX_WIDTH/2),pos_y + int(BOX_WIDTH/2)),
                               int(BOX_WIDTH/2) - box_padding, MOVE_WIDTH)
        elif color:
            pygame.draw.rect(screen, color, pygame.Rect((pos_x,pos_y), (BOX_WIDTH, BOX_WIDTH)))


    def draw_moves(game):
        for i in range(len(game.game_state)):
            if game.game_state[i]:
                draw_move(get_pos(i), game.game_state[i])


    def get_color(value):
        return (int((1.0-value) * 255.0), int(value * 255.0), 0)


    def color_boxes(agent):
        possible_move_infos = agent.get_possible_move_infos()
        for info in possible_move_infos:
            color = get_color(info[0])
            pos = get_pos(info[1])
            draw_move(pos, 3, color)


    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    screen.fill((0, 0, 0))
    draw_window(screen)

    agent = load_agent("meta_agent")
    #TODO: Make logger optional
    game_model = TicTacToe(logger)
    agent.enter(game_model)
    alpha = 0.2

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            if event.type == MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                box = get_box(pos)
                if box is not None and game_model.in_progress and box in game_model.possible_moves:
                    game_model.play_move(1,box)
                    print("Player made move {}", format(box))
                    draw_move(pos, 1)
                    if game_model.in_progress:
                        color_boxes(agent)
                        draw_window(screen)
                        pygame.display.flip()
                        time.sleep(0.5)
                        agent.play()
                        last_move = agent.get_last_move()
                        print("Agent made move {}", format(last_move))
                        print(game_model.game_state)
                        screen.fill((0, 0, 0))  # fill screen with black
                        draw_window(screen)
                        draw_moves(game_model)
                        pygame.display.flip()
                    if not game_model.in_progress:
                        pygame.display.flip()
                        time.sleep(0.5)
                        game_model = TicTacToe(logger)
                        if game_model.winner == 2:
                            agent.back_propagate_policies(alpha, 1.0)
                        elif game_model.winner == 1:
                            agent.back_propagate_policies(alpha, 0.0)
                        else:
                            agent.back_propagate_policies(alpha, 0.5)
                        agent.enter(game_model)
                        screen.fill((0, 0, 0))  # fill screen with black
                        draw_window(screen)
            pygame.event.clear()

        pygame.display.flip() # flip everything to the display
