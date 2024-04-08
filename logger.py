import collections



class Log:
    def __init__(self, max_steps, log_type, world_size):
        self.map_size = world_size
        self.steps = collections.deque(maxlen= max_steps)
        self.type = log_type

    def add_step(self, step):
        self.steps.append(step)

    


# MinMaxLoggers
class MinMax_Step:
    def __init__(self):
        self.world = None
        self.p1_tree = None
        self.p2_tree = None


    def set_world_state(self,game_state,food_bool):
        self.world = World_state(game_state,food_bool)



class MinMax_Node:
    def __init__(self):
        self.left_child = None
        self.center_child = None
        self.right_child = None
        self.heuristic = None
        self.choice = None

class MinMax_Decision_Tree:
    def __init__(self, depth):
        self.depth = depth
        self.root = None


class World_state:
    def __init__(self, game_state, food_bool):
        # P1 Grab
        self.p1_x = game_state.player1.x
        self.p1_y = game_state.player1.y
        self.p1_direction = game_state.player1.direction
        self.p1_tail = game_state.player1.tail
        self.p1_just_ate = game_state.player1.just_ate
        self.p1_last_tail = game_state.player1.last_Tail

        # P2 Grab
        self.p2_x = game_state.player2.x
        self.p2_y = game_state.player2.y
        self.p2_direction = game_state.player2.direction
        self.p2_tail = game_state.player2.tail
        self.p2_just_ate = game_state.player2.just_ate
        self.p2_last_tail = game_state.player2.last_Tail
        
        # World 
        self.food = game_state.food
        self.food_drawn = food_bool
