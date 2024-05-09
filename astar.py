import heapq
import math

# directions
UP = (0, -1)
RIGHT = (1, 0)
DOWN = (0, 1)
LEFT = (-1, 0)
# Use A*.
# Probably just make the opponent a wall, and put the 3 tiles next to the head as part of it, so we never try to move into a tile it might move into next.
class AStar_Node():
    def __init__(self):
        self.h = 0
        self.g = float('inf')
        self.f = float('inf')
        self.parent = (0, 0)

def heuristic(new_position, food_pos, opponent_tail, gs):
    distance_to_food = manhattan_distance(new_position, food_pos)

    #Penalty for being near other snake
    min_distance = min(manhattan_distance(new_position, pos) for pos in opponent_tail)
    proximity_penalty = max(0, (10 - min_distance))

    count = 0
    for direction in [UP, DOWN, LEFT, RIGHT]:
        np = ()
        count += 1

    opposite_directions = {
        (UP, DOWN), (DOWN, UP), (LEFT, RIGHT), (RIGHT, LEFT)
    }

    head_on_colision_penalty = 0
    if (gs.player1.direction, gs.player1.direction) in opposite_directions and manhattan_distance(new_position, opposite_directions[1]) == 2:
        head_on_colision_penalty = ('inf')


    return distance_to_food - 0.5 * proximity_penalty + head_on_colision_penalty


def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

#Calculates the euclidian distance between two points
def euclidean_distance(a, b):
    return math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))

def validate_successor(new_position, MAP_SIZE, explored_list, tail_locations):

    #Successor is not out of bounds
    if not (new_position[0] < MAP_SIZE[0] and new_position[0] >= 0
                and new_position[1] < MAP_SIZE[1] and new_position[1] >= 0):
        return False

    #Successor is not in a tail position
    if new_position in tail_locations:
        return False
    
    #Successor has not been visited before
    if explored_list[new_position[0]][new_position[1]] == True:
        return False
    
    return True

def get_path(explored_list_data, food_pos):
    path = []
    x = food_pos[0]
    y = food_pos[1]
    
    while not (explored_list_data[x][y].parent[0] == x and explored_list_data[x][y].parent[1] == y):
        path.insert(0, (x, y))
        next_x = explored_list_data[x][y].parent[0]
        next_y = explored_list_data[x][y].parent[1]
        x = next_x
        y = next_y
    
    path.insert(0, (x, y))
    return path

def a_star_search(start_pos, food_pos, tail_locations, MAP_SIZE, opponent_tail, gs):


    #Use an array for O(1) lookups
    explored_list = [[False for _ in range(MAP_SIZE[0])] for _ in range(MAP_SIZE[1])]
    explored_list_data = [[AStar_Node() for _ in range(MAP_SIZE[0])] for _ in range(MAP_SIZE[1])]

    #Init Start 
    start_node = explored_list_data[start_pos[0]][start_pos[1]]
    start_node.g = 0
    start_node.f = 0
    start_node.parent = (start_pos[0], start_pos[1])

    #Init frontier list
    frontier_list = []
    heapq.heappush(frontier_list, (0.0, start_pos))


    while len(frontier_list) > 0:

        #Pop next node
        popped_node = heapq.heappop(frontier_list)
        
        x = popped_node[1][0]
        y = popped_node[1][1]
        explored_list[x][y] = True

        #Enumerate successors
        for direction in [UP, DOWN, LEFT, RIGHT]:
            new_position = (x + direction[0], y + direction[1])

            if validate_successor(new_position, MAP_SIZE, explored_list, tail_locations):

                #Check if successors eats food
                current_successor_data = explored_list_data[new_position[0]][new_position[1]]
                if new_position == food_pos:

                    current_successor_data.parent = (x, y)
                    return get_path(explored_list_data, food_pos)
                    
                else:
                    new_g = explored_list_data[x][y].g + 1
                    new_h = heuristic(new_position, food_pos, opponent_tail, gs)
                    new_f = new_g + new_h

                    if (current_successor_data.f == float('inf') or 
                        current_successor_data.f > new_f):
                            
                            heapq.heappush(frontier_list, (new_f, new_position))

                            current_successor_data.g = new_g
                            current_successor_data.h = new_h
                            current_successor_data.f = new_f
                            current_successor_data.parent = (x, y)


def decide_move(gs, player, current_direction, tail_locations, MAP_SIZE, opponent_tail):
    start_pos = (player.x, player.y)

    #Calculate AStar if food generated
    if gs.food != (None, None):
        path = a_star_search(start_pos, gs.food, tail_locations, MAP_SIZE, opponent_tail, gs)

        if path == None:
            return ""
        
        if len(path) >= 1:
            next_node = path[1]
            next_move = (next_node[0] - start_pos[0], next_node[1] - start_pos[1])

            #Convert to direct ion
            if current_direction == next_move:
                return ""  
            elif (current_direction, next_move) in [(DOWN, LEFT), (LEFT, UP), (UP, RIGHT), (RIGHT, DOWN)]:
                return "RIGHT"
            elif (current_direction, next_move) in [(UP, LEFT), (LEFT, DOWN), (DOWN, RIGHT), (RIGHT, UP)]:
                return "LEFT"
            else:
                #Path chosen is backward into snake body
                #Default tiebreaker goes right for now
                return "RIGHT"
    return ""