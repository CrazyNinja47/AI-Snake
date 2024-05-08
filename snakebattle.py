import argparse
import sys
import os
import random
from pygame.locals import *
import minimax as minimax
import astar as astar
import copy
import logger as logger
import pickle

import neat
import math


FRAME_RATE = 60
MAP_SIZE = [30, 30]
TILE_SIZE = 20
START_LENGTH = 5

debug = False
logging = False

centered = True
headless = False
MAX_DEPTH = 4
# Log the last XX moves (each move includes both players)
LOG_LIMIT = 12
LOG_TYPE = "MinMax"
LOG_NAME = "log"

using_minimax_1 = False
using_minimax_2 = False
# Default P1
using_NEAT_P1 = False
using_NEAT_P2 = False
# Name of the pickle file with the NEAT NN
# It's not that impressive...
P1_NEATFILE = "NEATNeuralWinner.pkl"
P2_NEATFILE = "NEATNeuralWinner.pkl"

using_astar_1 = True
using_astar_2 = False


# Stores the GameState for use in AI.
# Needs to be able to determine future states for minimax.
class GameState:
    MAP_SIZE = MAP_SIZE

    def __init__(
        self, player1=None, player2=None, food=None, winner=None, food_drawn=False
    ):
        self.player1 = copy.deepcopy(player1)
        self.player2 = copy.deepcopy(player2)
        self.food = copy.deepcopy(food)
        self.winner = copy.deepcopy(winner)
        self.initialized = True
        self.food_drawn = food_drawn
        self.player1_full = False
        self.player2_full = False

    def get_square(self, x, y):
        # Is it our tail?
        for index, i in enumerate(self.player1.tail):
            if i[0] == x and i[1] == y:
                return 1
        # Is it their tail?
        for index, i in enumerate(self.player2.tail):
            if i[0] == x and i[1] == y:
                return 1
        # Is it outside?
        if x >= TILES_X or y >= TILES_Y or x < 0 or y < 0:
            return 1
        # Is it food?
        if x == self.food[0] and y == self.food[1]:
            return 2
        else:
            return 0

    ########  NEAT VISION METHODS  ########

    def get_left(self):
        if self.player1.direction[0] == 0:
            return (self.player1.direction[1], self.player1.direction[0])
        else:
            return (self.player1.direction[1] * (-1), self.player1.direction[0])

    def get_right(self):
        if self.player1.direction[0] == 0:
            return (self.player1.direction[1] * (-1), self.player1.direction[0])
        else:
            return (self.player1.direction[1], self.player1.direction[0])

    def append_onehot(self, code, arr):
        # SAFE  DEATH   FOOD
        # LEFT  MIDDLE  RIGHT
        if code == 0:
            arr.append(1)
            arr.append(0)
            arr.append(0)
        elif code == 1:
            arr.append(0)
            arr.append(1)
            arr.append(0)
        elif code == 2:
            arr.append(0)
            arr.append(0)
            arr.append(1)

    def append_food_side(self, arr):
        # player facing up or down
        # UP/DOWN Section
        if self.player1.direction[0] == 0:
            # player facing up (we check food's X)
            if self.player1.direction[1] == (-1):
                # food is to the left of the player
                if self.food[0] < self.player1.x:
                    self.append_onehot(0, arr)
                # food is directly in front
                elif self.food[0] == self.player1.x:
                    self.append_onehot(1, arr)
                # food is to the right of player
                elif self.food[0] > self.player1.x:
                    self.append_onehot(2, arr)
            # player facing down (we check food's X)
            if self.player1.direction[1] == (1):
                # food is to the right of the player
                if self.food[0] < self.player1.x:
                    self.append_onehot(2, arr)
                # food is directly in front
                elif self.food[0] == self.player1.x:
                    self.append_onehot(1, arr)
                # food is to the left of player
                elif self.food[0] > self.player1.x:
                    self.append_onehot(0, arr)

        # LEFT/RIGHT SECTION
        # Could just else:  but left this for readability
        elif self.player1.direction[1] == 0:
            # player facing left (we check food's Y)
            if self.player1.direction[0] == (-1):
                # food is to the right of the player
                if self.food[1] < self.player1.y:
                    self.append_onehot(2, arr)
                # food is directly in front
                elif self.food[1] == self.player1.y:
                    self.append_onehot(1, arr)
                # food is to the left of player
                elif self.food[1] > self.player1.y:
                    self.append_onehot(0, arr)
            # player facing right (we check food's Y)
            if self.player1.direction[0] == (1):
                # food is to the left of the player
                if self.food[1] < self.player1.y:
                    self.append_onehot(0, arr)
                # food is directly in front
                elif self.food[1] == self.player1.y:
                    self.append_onehot(1, arr)
                # food is to the right of player
                elif self.food[1] > self.player1.y:
                    self.append_onehot(2, arr)

    def snake_eyes(self):
        # {[P1 X], [P1 Y], [P2 X], [P2 Y], [Food X], [Food Y]}
        result = []
        # Tell snake which way to turn for food one hot-encoded
        #   0     1     2
        # Left Center Right
        # NOTE NOT SURE ABOUT THIS!
        # Get distance from food, elucidian
        # if self.food[0]:
        self.append_food_side(result)
        result.append(
            math.sqrt(
                ((self.player1.x - self.food[0]) ** 2)
                + ((self.player1.y - self.food[1]) ** 2)
            )
        )
        # else:
        # sometimes food doesn't exist, don't want to throw an error with a None
        # No food, go right
        # result.append(0)
        # result.append(0)
        # result.append(1)
        # # distance
        # result.append(0)
        # Wall distances - top and left
        result.append(self.player1.x)
        result.append(self.player1.y)
        # Wall distances - bottom and right
        result.append(abs(MAP_SIZE[0] - self.player1.x))
        result.append(abs(MAP_SIZE[1] - self.player1.y))
        # Adding vision cone, in widening radius
        # Going left to right
        vision = 2
        left = self.get_left()
        right = self.get_right()
        # Onehot Encode
        # EMPTY DANGER FOOD
        #   0      1    2
        for i in range(1, vision + 1):
            # left:
            spot = self.get_square(
                (self.player1.x + (left[0] * i)), (self.player1.x + (left[1] * i))
            )
            self.append_onehot(spot, result)
            # center:
            spot = self.get_square(
                (self.player1.x + (self.player1.direction[0] * i)),
                (self.player1.x + (self.player1.direction[1] * i)),
            )
            self.append_onehot(spot, result)
            spot = self.get_square(
                (self.player1.x + (right[0] * i)), (self.player1.x + (right[1] * i))
            )
            self.append_onehot(spot, result)
        return result

    ########  END OF NEAT VISION METHODS ########

    # def reset():
    #     self.Player1 = copy.deepcopy(self.saved_Player1)
    #     self.saved_Player2 = copy.deepcopy(self.saved_Player2)
    #     self.saved_Food = copy.deepcopy(self.saved_Food)
    #     self.winner = copy.deepcopy(self.saved_Winner)

    def update(self, player1, player2, food, winner, food_drawn):
        self.player1 = copy.deepcopy(player1)
        self.player2 = copy.deepcopy(player2)
        self.food = food
        self.winner = winner
        self.food_drawn = food_drawn

    def get_status(self):
        return GameState(
            copy.deepcopy(self.player1),
            copy.deepcopy(self.player2),
            copy.deepcopy(self.food),
            copy.deepcopy(self.winner),
            copy.deepcopy(self.food_drawn),
        )

    def is_terminal(self):
        return self.winner != None

    def to_string(self):
        return f"Player1: [{self.player1.x}, {self.player1.y}] | Player2: [{self.player2.x}, {self.player2.y}] | Food: [{self.food[0]}, {self.food[1]}] | Winner: {self.winner}"

    def next_state(self, state, move, player, moving):

        state.player1.just_ate = False
        state.player2.just_ate = False

        # We need to return game state after a movement step.
        # Probably can copy logic from below, perform a predicted step, and return the state.
        if moving == 1:
            moving_player = state.player1
        else:
            moving_player = state.player2

        if player == 1:
            target_player = state.player1
            target_win = 1
            opponent = state.player2
            opponent_win = 2
        else:
            target_player = state.player2
            target_win = 2
            opponent = state.player1
            opponent_win = 1
        #     temp = opp_direction
        #     opp_direction = self_direction
        #     self_direction = temp

        if move == "LEFT":
            moving_player.left = True
            moving_player.right = False
            moving_player.turn()
        elif move == "RIGHT":
            moving_player.left = False
            moving_player.right = True
            moving_player.turn()

        moving_player.left = False
        moving_player.right = False

        moving_player.x += moving_player.direction[0]
        moving_player.y += moving_player.direction[1]

        new_gs = state
        new_gs.winner = None

        moving_player.last_Tail = (
            moving_player.tail[moving_player.length - 1][0],
            (moving_player.tail[moving_player.length - 1][1]),
        )
        for i in range(moving_player.length - 1, -1, -1):
            if i == 0:
                moving_player.tail[i] = (moving_player.x, moving_player.y)
            else:
                moving_player.tail[i] = (
                    moving_player.tail[i - 1][0],
                    moving_player.tail[i - 1][1],
                )

        if moving_player.x == state.food[0] and moving_player.y == state.food[1]:
            state.food_drawn = False
            moving_player.tail.append(moving_player.last_Tail)
            moving_player.just_ate = True
            if moving == 1:
                self.player1_full = True
            else:
                self.player2_full = True
            # moving_player.tail.insert(
            #     0,
            #     (
            #         moving_player.x + moving_player.direction[0],
            #         moving_player.y + moving_player.direction[1],
            #     ),
            # )
            # moving_player.x = state.food[0] + moving_player.direction[0]
            # moving_player.y = state.food[1] + moving_player.direction[1]
            moving_player.length += 1

        if (
            target_player.x >= TILES_X
            or target_player.y >= TILES_Y
            or target_player.x < 0
            or target_player.y < 0
        ) and (
            opponent.x >= TILES_X
            or opponent.y >= TILES_Y
            or opponent.x < 0
            or opponent.y < 0
        ):
            new_gs.winner = 0
        else:
            if (
                target_player.x >= TILES_X
                or target_player.y >= TILES_Y
                or target_player.x < 0
                or target_player.y < 0
            ):
                if new_gs.winner == None:
                    new_gs.winner = opponent_win
                else:
                    new_gs.winner = 0
            elif (
                opponent.x >= TILES_X
                or opponent.y >= TILES_Y
                or opponent.x < 0
                or opponent.y < 0
            ):
                if new_gs.winner == None:
                    new_gs.winner = target_win
                else:
                    new_gs.winner = 0

        # check game over (touch)
        if (
            target_player.x == opponent.x
            and target_player.y == opponent.y
            and new_gs.winner is None
        ):
            new_gs.winner = 0
        else:
            # Check our tail
            # We killed them with our tail
            if new_gs.winner is None:
                for index, i in enumerate(target_player.tail):
                    # if debug: print(f'Checking P1: #{index} - ({i[0]},{i[1]})')
                    if i[0] == opponent.x and i[1] == opponent.y:
                        if new_gs.winner == None:
                            new_gs.winner = target_win
                            break
                        else:
                            new_gs.winner = 0
                            break
                    # We ate our own tail:
                    # if debug: print(f'Comparing P1 head ({target_player.x},{target_player.y}) to its tail: #{index} - ({i[0]},{i[1]})')
                    if (
                        index != 0
                        and i[0] == target_player.x
                        and i[1] == target_player.y
                    ):
                        if new_gs.winner == None:
                            new_gs.winner = opponent_win
                            break
                        else:
                            new_gs.winner = 0
                            break
            # Check Opponent Tail
            # They killed us with their tail
            if new_gs.winner is None:
                for index, i in enumerate(opponent.tail):
                    if i[0] == target_player.x and i[1] == target_player.y:
                        if new_gs.winner == None:
                            new_gs.winner = opponent_win
                            break
                        else:
                            new_gs.winner = 0
                            break
                        if index != 0 and i[0] == opponent.x and i[1] == opponent.y:
                            if new_gs.winner == None:
                                new_gs.winner = target_win
                                break
                            else:
                                new_gs.winner = 0
                                break
        return new_gs


gs = GameState()


# start arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    "--p1", type=str, help="AI type for Player 1 (minimax, NEAT, astar)"
)
arg_parser.add_argument(
    "--p2", type=str, help="AI type for Player 2 (minimax, NEAT, astar)"
)
arg_parser.add_argument("--headless", action="store_true", help="Run in headless mode")
arg_parser.add_argument(
    "-s",
    "--tilesize",
    dest="tilesize",
    metavar="PX",
    help="the size of a tile",
    type=int,
    default=TILE_SIZE,
)
arg_parser.add_argument(
    "-t",
    "--tiles",
    dest="tiles",
    nargs=2,
    metavar=("X", "Y"),
    help="the number of tiles",
    type=int,
    default=MAP_SIZE,
)
arg_parser.add_argument(
    "-d",
    "--debug",
    dest="debug",
    action="store_true",
    help="show debug information on the screen",
)
arg_parser.add_argument(
    "-f",
    "--fps",
    dest="fps",
    metavar="TPS",
    help="framerate in ticks per second",
    type=int,
    default=FRAME_RATE,
)
arg_parser.add_argument(
    "--pos_x", type=int, default=0, help="Horizontal position of the window"
)
arg_parser.add_argument(
    "--pos_y", type=int, default=0, help="Vertical position of the window"
)

args = arg_parser.parse_args()

# Set AI types based on command line arguments
if args.p1:
    using_minimax_1 = args.p1.lower() == "minimax"
    using_NEAT_P1 = args.p1.lower() == "neat"
    using_astar_1 = args.p1.lower() == "astar"

if args.p2:
    using_minimax_2 = args.p2.lower() == "minimax"
    using_NEAT_P2 = args.p2.lower() == "neat"
    using_astar_2 = args.p2.lower() == "astar"

# Set headless mode based on command line arguments
if args.headless:
    headless = True

if args.fps:
    FRAME_RATE = args.fps

import pygame

if headless:
    # center window
    os.environ["SDL_VIDEO_CENTERED"] = "1"
elif not centered:
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{args.pos_x},{args.pos_y}"
else:
    os.environ["SDL_VIDEO_CENTERED"] = "1"

# window dimensions
TILE_SIZE = args.tilesize
TILES_X = args.tiles[0]
TILES_Y = args.tiles[1]

# colors
COLOR_BG = (30, 30, 30)  # background
COLOR_FG = (255, 255, 255)  # foreground
COLOR_P1 = (255, 30, 30)  # player 1
COLOR_P2 = (30, 255, 30)  # player 2
COLOR_FD = (255, 200, 220)  # food
COLOR_DB = (50, 150, 250)  # debug

# settings
TPS = args.fps  # ticks lock
DEBUG = args.debug  # debugging


# tiles to pixels
def get_dimension(x, y, width=0, height=0):
    return (x * TILE_SIZE, y * TILE_SIZE, width * TILE_SIZE, height * TILE_SIZE)


if not headless:
    # init
    pygame.init()
    pygame.display.set_caption("Snake Battle by Scriptim")
    CLOCK = pygame.time.Clock()
    DISPLAY_SURFACE = pygame.display.set_mode(
        (TILE_SIZE * TILES_X, TILE_SIZE * TILES_Y)
    )
    DISPLAY_SURFACE.fill(COLOR_BG)

    # fonts (change font files here)
    FONT_DB = pygame.font.Font(None, 20)  # debug font
    FONT_SC = pygame.font.Font(None, TILE_SIZE * 5)  # score

# directions
UP = (0, -1)
RIGHT = (1, 0)
DOWN = (0, 1)
LEFT = (-1, 0)

# game over
winner = None


# print game over message
def game_over_msg(winner):
    if winner == 0:
        if p1.length == p2.length:
            if not headless:
                draw = FONT_SC.render("Draw!", 1, COLOR_FG)
                DISPLAY_SURFACE.blit(
                    draw,
                    (DISPLAY_SURFACE.get_width() / 2 - draw.get_rect().width / 2, 200),
                )
        elif p1.length > p2.length:
            game_over_msg(1)
        else:
            game_over_msg(2)
    elif winner == 1:
        if not headless:
            p1_wins = FONT_SC.render("Player 1 wins!", 1, COLOR_P1)
            DISPLAY_SURFACE.blit(
                p1_wins,
                (DISPLAY_SURFACE.get_width() / 2 - p1_wins.get_rect().width / 2, 200),
            )
    elif winner == 2:
        if not headless:
            p2_wins = FONT_SC.render("Player 2 wins!", 1, COLOR_P2)
            DISPLAY_SURFACE.blit(
                p2_wins,
                (DISPLAY_SURFACE.get_width() / 2 - p2_wins.get_rect().width / 2, 200),
            )


# food
food_drawn = False
food_x = None
food_y = None


# players
class Player:
    x = None
    y = None
    left = False
    right = False
    direction = None
    length = START_LENGTH
    tail = []

    def turn(self):
        if self.right:
            if self.direction == UP:
                self.direction = RIGHT
            elif self.direction == RIGHT:
                self.direction = DOWN
            elif self.direction == DOWN:
                self.direction = LEFT
            elif self.direction == LEFT:
                self.direction = UP
        elif self.left:
            if self.direction == UP:
                self.direction = LEFT
            elif self.direction == RIGHT:
                self.direction = UP
            elif self.direction == DOWN:
                self.direction = RIGHT
            elif self.direction == LEFT:
                self.direction = DOWN


p1 = Player()
p1.x = 4
p1.y = 10
p1.direction = DOWN
# p1.tail = [(p1.x, p1.y - 1), (p1.x, p1.y - 2)]
p1.tail = []
for i in range(1, p1.length + 1):
    p1.tail.append((p1.x, p1.y - i))
p1.last_Tail = None
p1.just_ate = False

p2 = Player()
p2.x = TILES_X - 5
p2.y = 10
p2.direction = DOWN
# p2.tail = [(p2.x, p2.y - 1), (p2.x, p2.y - 2)]
p2.tail = []
for i in range(1, p2.length + 1):
    p2.tail.append((p2.x, p2.y - i))
p2.last_Tail = None
p2.just_ate = False

main_log = logger.Log(LOG_LIMIT, LOG_TYPE, MAP_SIZE)

# Initialize food right off bat
safe_spot = False
while not safe_spot:
    food_x = random.choice(range(1, TILES_X - 1))
    food_y = random.choice(range(1, TILES_Y - 1))
    if (food_x != p1.x and food_y != p1.y) and (food_x != p2.x and food_y != p2.y):
        safe_spot = True
food_drawn = True


if using_NEAT_P1:
    genome = None
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "NEATSnakeConfig.txt")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    with open(P1_NEATFILE, "rb") as f:
        genome = pickle.load(f)
    f.close()
    net = neat.nn.FeedForwardNetwork.create(genome, config)

if using_NEAT_P2:
    genome = None
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "NEATSnakeConfig.txt")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    with open(P2_NEATFILE, "rb") as f:
        genome = pickle.load(f)
    f.close()
    net = neat.nn.FeedForwardNetwork.create(genome, config)

# main loop
while winner == None:
    current_step = logger.MinMax_Step()
    gs.update(
        player1=p1,
        player2=p2,
        food=(food_x, food_y),
        winner=winner,
        food_drawn=food_drawn,
    )
    p1.just_ate = False
    p2.just_ate = False
    # event queue
    if not headless:
        for event in pygame.event.get():
            # QUIT event
            if event.type == QUIT:
                print("## Quit ##")
                pygame.quit()
                sys.exit()
            # keyboard mode
            elif event.type == KEYDOWN:
                if event.key == K_DELETE:
                    # force draw
                    winner = 0
                if (event.key == K_a) and not using_minimax_1:
                    p1.left = True
                    p1.right = False
                    p1.turn()
                elif (event.key == K_d) and not using_minimax_1:
                    p1.right = True
                    p1.left = False
                    p1.turn()
                elif (event.key == K_LEFT) and not using_minimax_2:
                    p2.left = True
                    p2.right = False
                    p2.turn()
                elif (event.key == K_RIGHT) and not using_minimax_2:
                    p2.right = True
                    p2.left = False
                    p2.turn()

    if using_NEAT_P1:
        # print("NN:\n {}".format(genome))
        # print(f"{gs.snake_eyes()}")
        choices = net.activate(gs.snake_eyes())
        ourmax = max(choices)
        # print(f"Our choices: {choices}")
        move = choices.index(ourmax)
        # print(f"Snakeyes: picks {move} of {choices}")
        if move == 0:
            # Left
            p1.left = True
            p1.right = False
            p1.turn()
        elif move == 2:
            p1.left = False
            p1.right = True
            p1.turn()

    if using_NEAT_P2:
        # print("NN:\n {}".format(genome))
        # print(f"{gs.snake_eyes()}")
        choices = net.activate(gs.snake_eyes())
        ourmax = max(choices)
        # print(f"Our choices: {choices}")
        move = choices.index(ourmax)
        # print(f"Snakeyes: picks {move} of {choices}")
        if move == 0:
            # Left
            p2.left = True
            p2.right = False
            p2.turn()
        elif move == 2:
            p2.left = False
            p2.right = True
            p2.turn()

    if using_minimax_1:
        move = minimax.decide_move(gs, MAX_DEPTH, 1, current_step, logging)
        if move == "LEFT":
            p1.left = True
            p1.right = False
            p1.turn()
        elif move == "RIGHT":
            p1.left = False
            p1.right = True
            p1.turn()

    if using_minimax_2:
        move = minimax.decide_move(gs, MAX_DEPTH, 2, current_step, logging)
        if move == "LEFT":
            p2.left = True
            p2.right = False
            p2.turn()
        elif move == "RIGHT":
            p2.left = False
            p2.right = True
            p2.turn()

    if using_astar_1:
        move = astar.decide_move(
            gs, gs.player1, p1.direction, p1.tail + p2.tail, MAP_SIZE, p2.tail
        )
        if move == "LEFT":
            p1.left = True
            p1.right = False
            p1.turn()
        elif move == "RIGHT":
            p1.left = False
            p1.right = True
            p1.turn()

    if using_astar_2:
        move = astar.decide_move(
            gs, gs.player2, p2.direction, p1.tail + p2.tail, MAP_SIZE, p1.tail
        )
        if move == "LEFT":
            p2.left = True
            p2.right = False
            p2.turn()
        elif move == "RIGHT":
            p2.left = False
            p2.right = True
            p2.turn()

    p1.left = False
    p1.right = False
    p2.left = False
    p2.right = False

    if not headless:
        # clear
        DISPLAY_SURFACE.fill(COLOR_BG)
        # draw head
        pygame.draw.rect(DISPLAY_SURFACE, COLOR_P1, get_dimension(p1.x, p1.y, 1, 1))
        pygame.draw.rect(DISPLAY_SURFACE, COLOR_P2, get_dimension(p2.x, p2.y, 1, 1))

    # move head
    p1.x += p1.direction[0]
    p1.y += p1.direction[1]
    p2.x += p2.direction[0]
    p2.y += p2.direction[1]

    # move tail
    p1.last_Tail = (p1.tail[p1.length - 1][0], (p1.tail[p1.length - 1][1]))
    for i in range(p1.length - 1, -1, -1):
        if i == 0:
            p1.tail[i] = (p1.x, p1.y)
        else:
            p1.tail[i] = (p1.tail[i - 1][0], p1.tail[i - 1][1])

    p2.last_Tail = (p2.tail[p2.length - 1][0], (p2.tail[p2.length - 1][1]))
    for i in range(p2.length - 1, -1, -1):
        if i == 0:
            p2.tail[i] = (p2.x, p2.y)
        else:
            p2.tail[i] = (p2.tail[i - 1][0], p2.tail[i - 1][1])

    # draw tail
    if not headless:
        for i in p1.tail:
            pygame.draw.rect(DISPLAY_SURFACE, COLOR_P1, get_dimension(i[0], i[1], 1, 1))
        for i in p2.tail:
            pygame.draw.rect(DISPLAY_SURFACE, COLOR_P2, get_dimension(i[0], i[1], 1, 1))

    # food
    if food_drawn:
        if not headless:
            pygame.draw.rect(
                DISPLAY_SURFACE, COLOR_FD, get_dimension(food_x, food_y, 1, 1)
            )
        if p1.x == food_x and p1.y == food_y:
            p1.tail.append(p1.last_Tail)
            p1.just_ate = True
            # p1.tail.insert(0, (p1.x + p1.direction[0], p1.y + p1.direction[1]))
            # p1.x = food_x + p1.direction[0]
            # p1.y = food_y + p1.direction[1]
            p1.length += 1
            food_drawn = False
        elif p2.x == food_x and p2.y == food_y:
            p2.tail.append(p2.last_Tail)
            p2.just_ate = True
            # p2.tail.insert(0, (p2.x + p2.direction[0], p2.y + p2.direction[1]))
            # p2.x = food_x + p2.direction[0]
            # p2.y = food_y + p2.direction[1]
            p2.length += 1
            food_drawn = False
    else:
        # .95 normally
        if using_astar_1 or using_astar_2 or random.random() > 0:
            # prevent snake/food spawning
            safe_spot = False
            while not safe_spot:
                food_x = random.choice(range(1, TILES_X - 1))
                food_y = random.choice(range(1, TILES_Y - 1))
                if (food_x != p1.x and food_y != p1.y) and (
                    food_x != p2.x and food_y != p2.y
                ):
                    safe_spot = True
            food_drawn = True

    # score
    if not headless:
        p1_length_label = FONT_SC.render(str(p1.length), 1, COLOR_P1)
        sep_length_label = FONT_SC.render(":", 1, COLOR_FG)
        p2_length_label = FONT_SC.render(str(p2.length), 1, COLOR_P2)
        DISPLAY_SURFACE.blit(
            p1_length_label,
            (
                DISPLAY_SURFACE.get_width() / 2
                - p1_length_label.get_rect().width
                - TILE_SIZE,
                20,
            ),
        )
        DISPLAY_SURFACE.blit(sep_length_label, (DISPLAY_SURFACE.get_width() / 2, 20))
        DISPLAY_SURFACE.blit(
            p2_length_label,
            (
                DISPLAY_SURFACE.get_width() / 2
                + sep_length_label.get_rect().width
                + TILE_SIZE,
                20,
            ),
        )

    if logging:
        current_step.set_world_state(gs, food_drawn)
        main_log.add_step(current_step)

    # check game over (edges)
    if (p1.x >= TILES_X or p1.y >= TILES_Y or p1.x < 0 or p1.y < 0) and (
        p2.x >= TILES_X or p2.y >= TILES_Y or p2.x < 0 or p2.y < 0
    ):
        if debug:
            print(f"Tie:  Both off screen")
        winner = 0
    else:
        if p1.x >= TILES_X or p1.y >= TILES_Y or p1.x < 0 or p1.y < 0:
            if winner == None:
                if debug:
                    print(f"#2 wins:  1 went off screen")
                winner = 2
            else:
                if debug:
                    print(f"Tie:  1 off screen, 2 died...somehow?")
                winner = 0
        elif p2.x >= TILES_X or p2.y >= TILES_Y or p2.x < 0 or p2.y < 0:
            if winner == None:
                if debug:
                    print(f"1 Wins:  2 went off screen")
                winner = 1
            else:
                if debug:
                    print(f"Tie:  2 Off screen, 1 died....somehow?")
                winner = 0

    # check game over (touch)
    if p1.x == p2.x and p1.y == p2.y:
        if debug:
            print(f"Tie:  Head on collision")
        winner = 0
    else:
        for index, i in enumerate(p1.tail):
            if i[0] == p2.x and i[1] == p2.y:
                if winner == None:
                    if debug:
                        print(f"1 Wins:  2 ate P1 tail")
                    winner = 1
                    break
                else:
                    if debug:
                        print(f"Tie:  2 ate P1 tail....1 died...somehow?")
                    winner = 0
                    break
            if index != 0 and i[0] == p1.x and i[1] == p1.y:
                if winner == None:
                    if debug:
                        print(f"2 Wins:  1 ate own tail")
                    winner = 2
                    break
                else:
                    if debug:
                        print(f"Tie:  1 ate own tail, but 2 died somehow?")
                    winner = 0
                    break
        if winner is None:
            for index, i in enumerate(p2.tail):
                if i[0] == p1.x and i[1] == p1.y:
                    if winner == None:
                        if debug:
                            print(f"2 Wins:  1 ate P2 tail")
                        winner = 2
                        break
                    else:
                        winner = 0
                        break
                if index != 0 and i[0] == p2.x and i[1] == p2.y:
                    if winner == None:
                        winner = 1
                        break
                    else:
                        winner = 0
                        break

    if winner != None:
        print(f"Player {winner} won!")
        if headless:
            exit(0)
        game_over_msg(winner)
    # debugging
    if DEBUG:
        DISPLAY_SURFACE.blit(
            FONT_DB.render("Player 1", 1, COLOR_DB),
            (10, DISPLAY_SURFACE.get_height() - 80),
        )
        DISPLAY_SURFACE.blit(
            FONT_DB.render("Dir: " + str(p1.direction), 1, COLOR_DB),
            (10, DISPLAY_SURFACE.get_height() - 60),
        )
        DISPLAY_SURFACE.blit(
            FONT_DB.render("Length: " + str(p1.length), 1, COLOR_DB),
            (10, DISPLAY_SURFACE.get_height() - 40),
        )

        DISPLAY_SURFACE.blit(
            FONT_DB.render("Player 2", 1, COLOR_DB),
            (180, DISPLAY_SURFACE.get_height() - 80),
        )
        DISPLAY_SURFACE.blit(
            FONT_DB.render("Dir: " + str(p2.direction), 1, COLOR_DB),
            (180, DISPLAY_SURFACE.get_height() - 60),
        )
        DISPLAY_SURFACE.blit(
            FONT_DB.render("Length: " + str(p2.length), 1, COLOR_DB),
            (180, DISPLAY_SURFACE.get_height() - 40),
        )

        DISPLAY_SURFACE.blit(
            FONT_DB.render("MS: " + str(pygame.time.get_ticks()), 1, COLOR_DB),
            (340, DISPLAY_SURFACE.get_height() - 80),
        )
        DISPLAY_SURFACE.blit(
            FONT_DB.render("FPS: " + str(round(CLOCK.get_fps(), 2)), 1, COLOR_DB),
            (340, DISPLAY_SURFACE.get_height() - 60),
        )
        if food_drawn:
            DISPLAY_SURFACE.blit(
                FONT_DB.render(
                    "Food: (" + str(food_x) + ", " + str(food_y) + ")", 1, COLOR_DB
                ),
                (340, DISPLAY_SURFACE.get_height() - 40),
            )
        else:
            DISPLAY_SURFACE.blit(
                FONT_DB.render("Food: -", 1, COLOR_DB),
                (340, DISPLAY_SURFACE.get_height() - 40),
            )

    # update
    if not headless:
        CLOCK.tick(TPS)
        # Save current world state to logging if true
        pygame.display.update()


if logging:
    gs.update(
        player1=p1,
        player2=p2,
        food=(food_x, food_y),
        winner=winner,
        food_drawn=food_drawn,
    )
    current_step.set_world_state(gs, food_drawn)
    main_log.add_step(current_step)

    with open(LOG_NAME + ".pkl", "wb") as f:
        pickle.dump(main_log, f)
    f.close()
pygame.time.wait(4000)
