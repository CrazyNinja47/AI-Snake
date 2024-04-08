import argparse
import pygame
import sys
import os
import random
from pygame.locals import *
import minimax as minimax
import copy
import logger as logger
import pickle



FRAME_RATE = 30
MAP_SIZE = [15  , 15]
TILE_SIZE = 10
START_LENGTH = 5

debug = False
logging = True


MAX_DEPTH = 5
# Log the last XX moves (each move includes both players)
LOG_LIMIT = 12
LOG_TYPE = "MinMax"
LOG_NAME = "log"

using_minimax_1 = True
using_minimax_2 = True



# Stores the GameState for use in AI.
# Needs to be able to determine future states for minimax.
class GameState:
    MAP_SIZE = MAP_SIZE
    def __init__(self, player1=None, player2=None, food=None, winner=None, food_drawn =False):
        self.player1 = copy.deepcopy(player1)
        self.player2 = copy.deepcopy(player2)
        self.food = copy.deepcopy(food)
        self.winner = copy.deepcopy(winner)
        self.initialized = True
        self.food_drawn = food_drawn


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
            copy.deepcopy(self.food_drawn)
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

        moving_player.last_Tail = (moving_player.tail[moving_player.length -1 ][0],(moving_player.tail[moving_player.length -1][1]))
        for i in range(moving_player.length - 1, -1, -1):
            if i == 0:
                moving_player.tail[i] = (moving_player.x, moving_player.y)
            else:
                moving_player.tail[i] = (
                    moving_player.tail[i - 1][0],
                    moving_player.tail[i - 1][1],
                )

        if moving_player.x == state.food[0] and moving_player.y == state.food[1]:
            moving_player.tail.append(moving_player.last_Tail)
            moving_player.just_ate = True
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
        if (target_player.x == opponent.x and target_player.y == opponent.y and new_gs.winner is None):
            new_gs.winner = 0
        else:
            # Check our tail
            # We killed them with our tail
            if new_gs.winner is None:
                for index, i in enumerate(target_player.tail):
                    #if debug: print(f'Checking P1: #{index} - ({i[0]},{i[1]})')
                    if i[0] == opponent.x and i[1] == opponent.y:
                        if new_gs.winner == None:
                            new_gs.winner = target_win
                            break
                        else:
                            new_gs.winner = 0
                            break   
            # We ate our own tail:
                    #if debug: print(f'Comparing P1 head ({target_player.x},{target_player.y}) to its tail: #{index} - ({i[0]},{i[1]})')
                    if index != 0 and i[0] == target_player.x and i[1] == target_player.y:
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
    "-r",
    "--raspi",
    dest="raspi",
    action="store_true",
    help="run snake battle on a raspberry pi",
)
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
    "-b",
    "--delay",
    dest="delay",
    metavar="MS",
    help="button delay (raspi mode)",
    type=int,
    default=100,
)
args = arg_parser.parse_args()
print(args)

# center window
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


# init
pygame.init()
pygame.display.set_caption("Snake Battle by Scriptim")
CLOCK = pygame.time.Clock()
DISPLAY_SURFACE = pygame.display.set_mode((TILE_SIZE * TILES_X, TILE_SIZE * TILES_Y))
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
            draw = FONT_SC.render("Draw!", 1, COLOR_FG)
            DISPLAY_SURFACE.blit(
                draw, (DISPLAY_SURFACE.get_width() / 2 - draw.get_rect().width / 2, 200)
            )
        elif p1.length > p2.length:
            game_over_msg(1)
        else:
            game_over_msg(2)
    elif winner == 1:
        p1_wins = FONT_SC.render("Player 1 wins!", 1, COLOR_P1)
        DISPLAY_SURFACE.blit(
            p1_wins,
            (DISPLAY_SURFACE.get_width() / 2 - p1_wins.get_rect().width / 2, 200),
        )
    elif winner == 2:
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

main_log = logger.Log(LOG_LIMIT,LOG_TYPE, MAP_SIZE)

# main loop
while winner == None:
    current_step = logger.MinMax_Step()
    gs.update(player1=p1, player2=p2, food=(food_x, food_y), winner=winner, food_drawn=food_drawn)
    p1.just_ate = False
    p2.just_ate = False
    # event queue
    for event in pygame.event.get():
        # QUIT event
        if event.type == QUIT:
            print("## Quit ##")
            pygame.quit()
            sys.exit()
        # keyboard mode
        elif event.type == KEYDOWN and not args.raspi:
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

    p1.left = False
    p1.right = False
    p2.left = False
    p2.right = False


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
    p1.last_Tail = (p1.tail[p1.length -1 ][0],(p1.tail[p1.length -1][1]))
    for i in range(p1.length - 1, -1, -1):
        if i == 0:
            p1.tail[i] = (p1.x, p1.y)
        else:
            p1.tail[i] = (p1.tail[i - 1][0], p1.tail[i - 1][1])

    p2.last_Tail = (p2.tail[p2.length -1][0],(p2.tail[p2.length -1][1]))
    for i in range(p2.length - 1, -1, -1):
        if i == 0:
            p2.tail[i] = (p2.x, p2.y)
        else:
            p2.tail[i] = (p2.tail[i - 1][0], p2.tail[i - 1][1])


    # draw tail
    for i in p1.tail:
        pygame.draw.rect(DISPLAY_SURFACE, COLOR_P1, get_dimension(i[0], i[1], 1, 1))
    for i in p2.tail:
        pygame.draw.rect(DISPLAY_SURFACE, COLOR_P2, get_dimension(i[0], i[1], 1, 1))



    # food
    if food_drawn:
        pygame.draw.rect(DISPLAY_SURFACE, COLOR_FD, get_dimension(food_x, food_y, 1, 1))
        if p1.x == food_x and p1.y == food_y:
            p1.tail.append(p1.last_Tail)
            p1.just_ate = True
            #p1.tail.insert(0, (p1.x + p1.direction[0], p1.y + p1.direction[1]))
            # p1.x = food_x + p1.direction[0]
            # p1.y = food_y + p1.direction[1]
            p1.length += 1
            food_drawn = False
        elif p2.x == food_x and p2.y == food_y:
            p2.tail.append(p2.last_Tail)
            p2.just_ate = True
            #p2.tail.insert(0, (p2.x + p2.direction[0], p2.y + p2.direction[1]))
            # p2.x = food_x + p2.direction[0]
            # p2.y = food_y + p2.direction[1]
            p2.length += 1
            food_drawn = False
    else:
        if random.random() > 0.95:
            food_x = random.choice(range(1, TILES_X - 1))
            food_y = random.choice(range(1, TILES_Y - 1))
            food_drawn = True

    # score
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
        if debug: print(f'Tie:  Both off screen')
        winner = 0
    else:
        if p1.x >= TILES_X or p1.y >= TILES_Y or p1.x < 0 or p1.y < 0:
            if winner == None:
                if debug: print(f'#2 wins:  1 went off screen')
                winner = 2
            else:
                if debug: print(f'Tie:  1 off screen, 2 died...somehow?')
                winner = 0
        elif p2.x >= TILES_X or p2.y >= TILES_Y or p2.x < 0 or p2.y < 0:
            if winner == None:
                if debug: print(f'1 Wins:  2 went off screen')
                winner = 1
            else:
                if debug: print(f'Tie:  2 Off screen, 1 died....somehow?')
                winner = 0

    # check game over (touch)
    if p1.x == p2.x and p1.y == p2.y:
        if debug: print(f'Tie:  Head on collision')
        winner = 0
    else:
        for index, i in enumerate(p1.tail):
            if i[0] == p2.x and i[1] == p2.y:
                if winner == None:
                    if debug: print(f'1 Wins:  2 ate P1 tail')
                    winner = 1
                    break
                else:
                    if debug: print(f'Tie:  2 ate P1 tail....1 died...somehow?')
                    winner = 0
                    break
            if index != 0 and i[0] == p1.x and i[1] == p1.y:
                if winner == None:
                    if debug: print(f'2 Wins:  1 ate own tail')
                    winner = 2
                    break
                else:
                    if debug: print(f'Tie:  1 ate own tail, but 2 died somehow?')
                    winner = 0
                    break
        if winner is None:
            for index, i in enumerate(p2.tail):
                if i[0] == p1.x and i[1] == p1.y:
                    if winner == None:
                        if debug: print(f'2 Wins:  1 ate P2 tail')
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
        print(f'Player {winner} won!')
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
    CLOCK.tick(TPS)
    # Save current world state to logging if true
    pygame.display.update()


if logging:
    gs.update(player1=p1, player2=p2, food=(food_x, food_y), winner=winner, food_drawn=food_drawn)
    current_step.set_world_state(gs, food_drawn)
    main_log.add_step(current_step)
    
    with open(LOG_NAME + '.pkl', "wb") as f:
        pickle.dump(main_log, f)
    f.close()
pygame.time.wait(4000)
