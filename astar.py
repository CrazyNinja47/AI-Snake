import time
import math

# directions
UP = (0, -1)
RIGHT = (1, 0)
DOWN = (0, 1)
LEFT = (-1, 0)
# Use A*.
# Probably just make the opponent a wall, and put the 3 tiles next to the head as part of it, so we never try to move into a tile it might move into next.
class AStar_Node():
    def __init__(self, parent=None, position=None):
        self.h = 0
        self.g = 0
        self.f = 0
        self.parent = parent
        self.position = position
    
    def __eq__(self, o):
        return self.position == o.position

#Heuristic uses the euclidian distance between food and snake"
def heuristic(a, b):
    return math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))

def a_star_search(start_pos, food_pos, tail_locations):

    frontier_list = []
    explored_list = []

    start_node = AStar_Node(None, start_pos)
    end_node = AStar_Node(None, food_pos)

    frontier_list.append(start_node)

    while len(frontier_list) > 0:
        
        #NEEDS TO BE FIXED
        #When food is picked up something causes the parent loop To go infinite   
        #while len(frontier_list) > 0:      
        #To keep program runnning it returns after frontier_list is too big
        if len(frontier_list) > 1000:
            return []

        #Get Current Node"
        current_node = frontier_list[0]
        current_index = 0

        for index, item in enumerate(frontier_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        #Remove current 
        frontier_list.pop(current_index)
        explored_list.append(current_node)

        #Check If Food Reached
        if current_node == end_node:
            path = []
            cur = current_node

            while cur is not None:
                path.append(cur.position)
                cur = cur.parent
            return path[::-1]


        #Generate Adjacent Nodes
        adj_nodes = []
        for new_position in [UP, DOWN, LEFT, RIGHT]:
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])
            
            #Check if node is a tail
            if node_position in tail_locations:
                continue

            new_node = AStar_Node(current_node, node_position)
            adj_nodes.append(new_node)


        #Iteratre Adjacent Nodes
        for n in adj_nodes:

            #Node has been visted before
            if n in explored_list:
                continue
            
            n.g = current_node.g + 1
            n.h = heuristic(n.position, end_node.position)
            n.f = n.g + n.h

            #Node has been visted before
            for x in frontier_list:
                if n == x and n.g > x.g:
                    continue

            frontier_list.append(n)

def decide_move(gs, player, current_direction, tail_locations):
    start_pos = (player.x, player.y)

    #Calculate AStar if food generated
    if gs.food != (None, None):
        path = a_star_search(start_pos, gs.food, tail_locations)
        
        if len(path) > 1:
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