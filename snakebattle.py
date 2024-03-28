import argparse, pygame, sys, os, random
from pygame.locals import *
import minimax as minimax
import copy

max_depth = 3

using_minimax = True


# Stores the GameState for use in AI.
# Needs to be able to determine future states for minimax.
class GameState:

    def __init__(self, player1=None, player2=None, food=None, winner=None):
        self.player1 = copy.deepcopy(player1)
        self.player2 = copy.deepcopy(player2)
        self.food = food
        self.winner = winner
        self.initialized = True

    def update(self, player1, player2, food, winner):
        self.player1 = copy.deepcopy(player1)
        self.player2 = copy.deepcopy(player2)
        self.food = food
        self.winner = winner

    def get_children(self):
        directions = ["STRAIGHT", "LEFT", "RIGHT"]
        children_with_moves = []

        for opp_direction in directions:
            for self_direction in directions:
                new_state = self.next_state(self_direction, opp_direction)
                children_with_moves.append((new_state, self_direction))

        return children_with_moves

    def is_terminal(self):
        return self.winner != None

    def to_string(self):
        return f"Player1: [{self.player1.x}, {self.player1.y}] | Player2: [{self.player2.x}, {self.player2.y}] | Food: [{self.food[0]}, {self.food[1]}] | Winner: {self.winner}"

    def next_state(self, self_direction, opp_direction):
        # We need to return game state after a movement step.
        # Probably can copy logic from below, perform a predicted step, and return the state.
        new_gs = GameState(
            copy.deepcopy(self.player1),
            copy.deepcopy(self.player2),
            self.food,
            self.winner,
        )
        if self_direction == "LEFT":
            new_gs.player2.left = True
            new_gs.player2.right = False
            new_gs.player2.turn()
        elif self_direction == "RIGHT":
            new_gs.player2.left = False
            new_gs.player2.right = True
            new_gs.player2.turn()

        if opp_direction == "LEFT":
            new_gs.player1.left = True
            new_gs.player1.right = False
            new_gs.player1.turn()
        elif opp_direction == "RIGHT":
            new_gs.player1.left = False
            new_gs.player1.right = True
            new_gs.player1.turn()

        new_gs.player1.left = False
        new_gs.player1.right = False
        new_gs.player2.left = False
        new_gs.player2.right = False

        new_gs.player1.x += new_gs.player1.direction[0]
        new_gs.player1.y += new_gs.player1.direction[1]
        new_gs.player2.x += new_gs.player2.direction[0]
        new_gs.player2.y += new_gs.player2.direction[1]

        for i in range(new_gs.player1.length - 1, -1, -1):
            if i == 0:
                new_gs.player1.tail[i] = (new_gs.player1.x, new_gs.player1.y)
            else:
                new_gs.player1.tail[i] = (
                    new_gs.player1.tail[i - 1][0],
                    new_gs.player1.tail[i - 1][1],
                )
        for i in range(new_gs.player2.length - 1, -1, -1):
            if i == 0:
                new_gs.player2.tail[i] = (new_gs.player2.x, new_gs.player2.y)
            else:
                new_gs.player2.tail[i] = (
                    new_gs.player2.tail[i - 1][0],
                    new_gs.player2.tail[i - 1][1],
                )

        if new_gs.player1.x == new_gs.food[0] and new_gs.player1.y == new_gs.food[1]:
            new_gs.player1.tail.insert(
                0,
                (
                    new_gs.player1.x + new_gs.player1.direction[0],
                    new_gs.player1.y + new_gs.player1.direction[1],
                ),
            )
            new_gs.player1.x = new_gs.food[0] + new_gs.player1.direction[0]
            new_gs.player1.y = new_gs.food[1] + new_gs.player1.direction[1]
            new_gs.player1.length += 1

        elif new_gs.player2.x == new_gs.food[0] and new_gs.player2.y == new_gs.food[1]:
            new_gs.player2.tail.insert(
                0,
                (
                    new_gs.player2.x + new_gs.player2.direction[0],
                    new_gs.player2.y + new_gs.player2.direction[1],
                ),
            )
            new_gs.player2.x = new_gs.food[0] + new_gs.player2.direction[0]
            new_gs.player2.y = new_gs.food[1] + new_gs.player2.direction[1]
            new_gs.player2.length += 1

        if (
            new_gs.player1.x >= TILES_X
            or new_gs.player1.y >= TILES_Y
            or new_gs.player1.x < 0
            or new_gs.player1.y < 0
        ) and (
            new_gs.player2.x >= TILES_X
            or new_gs.player2.y >= TILES_Y
            or new_gs.player2.x < 0
            or new_gs.player2.y < 0
        ):
            new_gs.winner = 0
        else:
            if (
                new_gs.player1.x >= TILES_X
                or new_gs.player1.y >= TILES_Y
                or new_gs.player1.x < 0
                or new_gs.player1.y < 0
            ):
                if new_gs.winner == None:
                    new_gs.winner = 2
                else:
                    new_gs.winner = 0
            elif (
                new_gs.player2.x >= TILES_X
                or new_gs.player2.y >= TILES_Y
                or new_gs.player2.x < 0
                or new_gs.player2.y < 0
            ):
                if new_gs.winner == None:
                    new_gs.winner = 1
                else:
                    new_gs.winner = 0

        ## check game over (touch)
        if (
            new_gs.player1.x == new_gs.player2.x
            and new_gs.player1.y == new_gs.player2.y
        ):
            new_gs.winner = 0
        else:
            for i in new_gs.player1.tail:
                if i[0] == new_gs.player2.x and i[1] == new_gs.player2.y:
                    if new_gs.winner == None:
                        new_gs.winner = 1
                    else:
                        new_gs.winner = 0
            for i in new_gs.player2.tail:
                if i[0] == new_gs.player1.x and i[1] == new_gs.player1.y:
                    if new_gs.winner == None:
                        new_gs.winner = 2
                    else:
                        new_gs.winner = 0

        # print(new_gs.to_string())
        return new_gs


gs = GameState()

## start arguments
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
    default=24,
)
arg_parser.add_argument(
    "-t",
    "--tiles",
    dest="tiles",
    nargs=2,
    metavar=("X", "Y"),
    help="the number of tiles",
    type=int,
    default=[70, 50],
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
    default=12,
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

## center window
os.environ["SDL_VIDEO_CENTERED"] = "1"


## window dimensions
TILE_SIZE = args.tilesize
TILES_X = args.tiles[0]
TILES_Y = args.tiles[1]

## colors
COLOR_BG = (30, 30, 30)  # background
COLOR_FG = (255, 255, 255)  # foreground
COLOR_P1 = (255, 30, 30)  # player 1
COLOR_P2 = (30, 255, 30)  # player 2
COLOR_FD = (255, 200, 30)  # food
COLOR_DB = (50, 150, 250)  # debug

## settings
TPS = args.fps  # ticks lock
DEBUG = args.debug  # debugging


## tiles to pixels
def get_dimension(x, y, width=0, height=0):
    return (x * TILE_SIZE, y * TILE_SIZE, width * TILE_SIZE, height * TILE_SIZE)


## init
pygame.init()
pygame.display.set_caption("Snake Battle by Scriptim")
CLOCK = pygame.time.Clock()
DISPLAY_SURFACE = pygame.display.set_mode((TILE_SIZE * TILES_X, TILE_SIZE * TILES_Y))
DISPLAY_SURFACE.fill(COLOR_BG)

## fonts (change font files here)
FONT_DB = pygame.font.Font(None, 20)  # debug font
FONT_SC = pygame.font.Font(None, TILE_SIZE * 5)  # score

## directions
UP = (0, -1)
RIGHT = (-1, 0)
DOWN = (0, 1)
LEFT = (1, 0)

## game over
winner = None


## print game over message
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


## food
food_drawn = False
food_x = None
food_y = None


## players
class Player:
    x = None
    y = None
    left = False
    right = False
    direction = None
    length = 2
    tail = []

    def turn(self):
        if self.right:
            if self.direction == UP:
                self.direction = LEFT
            elif self.direction == RIGHT:
                self.direction = UP
            elif self.direction == DOWN:
                self.direction = RIGHT
            elif self.direction == LEFT:
                self.direction = DOWN
        elif self.left:
            if self.direction == UP:
                self.direction = RIGHT
            elif self.direction == RIGHT:
                self.direction = DOWN
            elif self.direction == DOWN:
                self.direction = LEFT
            elif self.direction == LEFT:
                self.direction = UP


p1 = Player()
p1.x = 4
p1.y = TILES_Y / 2 + 5
p1.direction = UP
p1.tail = [(p1.x, p1.y - 1), (p1.x, p1.y - 2)]

p2 = Player()
p2.x = TILES_X - 5
p2.y = TILES_Y / 2 + 5
p2.direction = UP
p2.tail = [(p2.x, p2.y - 1), (p2.x, p2.y - 2)]

## main loop
while winner == None:
    gs.update(player1=p1, player2=p2, food=(food_x, food_y), winner=winner)
    ## event queue
    for event in pygame.event.get():
        ## QUIT event
        if event.type == QUIT:
            print("## Quit ##")
            pygame.quit()
            sys.exit()
        ## keyboard mode
        elif event.type == KEYDOWN and not args.raspi:
            if event.key == K_a:
                p1.left = True
                p1.right = False
                p1.turn()
            elif event.key == K_d:
                p1.right = True
                p1.left = False
                p1.turn()
            elif (event.key == K_LEFT) and not using_minimax:
                p2.left = True
                p2.right = False
                p2.turn()
            elif (event.key == K_RIGHT) and not using_minimax:
                p2.right = True
                p2.left = False
                p2.turn()
    if minimax:
        move = minimax.decide_move(gs, max_depth)
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

    ## clear
    DISPLAY_SURFACE.fill(COLOR_BG)

    ## draw head
    pygame.draw.rect(DISPLAY_SURFACE, COLOR_P1, get_dimension(p1.x, p1.y, 1, 1))
    pygame.draw.rect(DISPLAY_SURFACE, COLOR_P2, get_dimension(p2.x, p2.y, 1, 1))

    ## move head
    p1.x += p1.direction[0]
    p1.y += p1.direction[1]
    p2.x += p2.direction[0]
    p2.y += p2.direction[1]

    ## draw tail
    for i in p1.tail:
        pygame.draw.rect(DISPLAY_SURFACE, COLOR_P1, get_dimension(i[0], i[1], 1, 1))
    for i in p2.tail:
        pygame.draw.rect(DISPLAY_SURFACE, COLOR_P2, get_dimension(i[0], i[1], 1, 1))

    ## move tail
    for i in range(p1.length - 1, -1, -1):
        if i == 0:
            p1.tail[i] = (p1.x, p1.y)
        else:
            p1.tail[i] = (p1.tail[i - 1][0], p1.tail[i - 1][1])
    for i in range(p2.length - 1, -1, -1):
        if i == 0:
            p2.tail[i] = (p2.x, p2.y)
        else:
            p2.tail[i] = (p2.tail[i - 1][0], p2.tail[i - 1][1])

    ## food
    if food_drawn:
        pygame.draw.rect(DISPLAY_SURFACE, COLOR_FD, get_dimension(food_x, food_y, 1, 1))
        if p1.x == food_x and p1.y == food_y:
            p1.tail.insert(0, (p1.x + p1.direction[0], p1.y + p1.direction[1]))
            p1.x = food_x + p1.direction[0]
            p1.y = food_y + p1.direction[1]
            p1.length += 1
            food_drawn = False
        elif p2.x == food_x and p2.y == food_y:
            p2.tail.insert(0, (p2.x + p2.direction[0], p2.y + p2.direction[1]))
            p2.x = food_x + p2.direction[0]
            p2.y = food_y + p2.direction[1]
            p2.length += 1
            food_drawn = False
    else:
        if random.random() > 0.95:
            food_x = random.choice(range(1, TILES_X - 1))
            food_y = random.choice(range(1, TILES_Y - 1))
            food_drawn = True

    ## score
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

    ## check game over (edges)
    if (p1.x >= TILES_X or p1.y >= TILES_Y or p1.x < 0 or p1.y < 0) and (
        p2.x >= TILES_X or p2.y >= TILES_Y or p2.x < 0 or p2.y < 0
    ):
        winner = 0
    else:
        if p1.x >= TILES_X or p1.y >= TILES_Y or p1.x < 0 or p1.y < 0:
            if winner == None:
                winner = 2
            else:
                winner = 0
        elif p2.x >= TILES_X or p2.y >= TILES_Y or p2.x < 0 or p2.y < 0:
            if winner == None:
                winner = 1
            else:
                winner = 0

    ## check game over (touch)
    if p1.x == p2.x and p1.y == p2.y:
        winner = 0
    else:
        for i in p1.tail:
            if i[0] == p2.x and i[1] == p2.y:
                if winner == None:
                    winner = 1
                else:
                    winner = 0
        for i in p2.tail:
            if i[0] == p1.x and i[1] == p1.y:
                if winner == None:
                    winner = 2
                else:
                    winner = 0

    if winner != None:
        game_over_msg(winner)

    ## debugging
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

    ## update
    CLOCK.tick(TPS)
    pygame.display.update()


pygame.time.wait(4000)
