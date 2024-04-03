# Main issue with minimax is we need to get future states, to be added to GameState
import stateUtils as stateUtils
import copy as copy
LENGTH_WEIGHT = 50
AGGRO_WEIGHT = 5
FOOD_DIST_WEIGHT = 20
WIN_WEIGHT = 1000
LOSE_WEIGHT = 1000
DRAW_WEIGHT = 500
EDGE_WEIGHT = 10

debug = False



def heuristic(game_state, player):
 
    test = {'length':0, 'aggro':0, 'food':0, 'win':0, 'edge':0}

    player1 = game_state.player1
    player2 = game_state.player2

    food_x, food_y = game_state.food
    winner = game_state.winner

    score = 0

    if player == 2:
        score += player2.length * LENGTH_WEIGHT
        score -= player1.length * LENGTH_WEIGHT
    else:
        score -= player2.length * LENGTH_WEIGHT
        score += player1.length * LENGTH_WEIGHT
    
    test.update(({'length':score}))

    player_distance = abs(player1.x - player2.x) + abs(player1.y - player2.y)

    # Reward for moving perpendicular to opponent
    # if player2.direction in ["UP", "DOWN"] and player1.direction in ["LEFT", "RIGHT"]:
    #     score += 10
    # elif player2.direction in ["LEFT", "RIGHT"] and player1.direction in ["UP", "DOWN"]:
    #     score += 10

    aggro_score = 0
    # Reward for being close to the opponent as it gets longer
    if player_distance > 0:
        aggro_score += 1 / player_distance
        if player == 2:
            aggro_score *= (player2.length - player1.length) * AGGRO_WEIGHT
        else:
            aggro_score *= (player1.length - player2.length) * AGGRO_WEIGHT
        # If there is no food on the map, act much more aggressively - otherwise it will just go in circles
        if (food_x is None) or (food_y is None):
            aggro_score = (aggro_score + 1) * AGGRO_WEIGHT

    score += aggro_score
    test.update(({'aggro':aggro_score}))
    food_distance = 0

    # Penalize for moving away from food
    if (food_x is not None) and (food_y is not None):
        if player == 2:
            food_distance = abs(player2.x - food_x) + abs(player2.y - food_y)
        else:
            food_distance = abs(player1.x - food_x) + abs(player1.y - food_y)
        dis_score = food_distance * FOOD_DIST_WEIGHT
    else:
        dis_score = 0
    test.update(({'food':-(dis_score)}))
    score -= dis_score
    

    # Reward for winning or losing
    if player == 2:
        if winner == 1:
            test.update(({'win':-(LOSE_WEIGHT)}))
            score -= LOSE_WEIGHT
        if winner == 2:
            score += WIN_WEIGHT
            test.update(({'win':WIN_WEIGHT}))
        if winner == 0:
            score -= DRAW_WEIGHT 
            test.update(({'win':-(DRAW_WEIGHT)}))
    if player == 1:
        if winner == 2:
            score -= LOSE_WEIGHT
            test.update(({'win':-LOSE_WEIGHT}))
        if winner == 1:
            test.update(({'win':WIN_WEIGHT}))
            score += WIN_WEIGHT
        if winner == 0:
            score -= DRAW_WEIGHT 
            test.update(({'win':-(DRAW_WEIGHT)}))
    
    if player == 1:
        playerID = player1
        enemyID = player2
    else:
        playerID = player2
        enemyID = player1


    # if playerID.x < 1 or playerID.x >= (game_state.MAP_SIZE[0] - 2 ):
    #     score -= EDGE_WEIGHT
    # if playerID.y < 1 or playerID.y >= (game_state.MAP_SIZE[1] - 2 ):
    #     score -= EDGE_WEIGHT
    # if enemyID.x < 1 or enemyID.x >= (game_state.MAP_SIZE[0] - 2 ):
    #     score += EDGE_WEIGHT
    # if enemyID.y < 1 or enemyID.y >= (game_state.MAP_SIZE[1] - 2 ):
    #     score += EDGE_WEIGHT




    if debug: print(f'#**# (Heuristic) Player {player} @ ({playerID.x},{playerID.y}) with {test} and total: {round(sum(test.values()),4)} #**#')
    return score


def minimax(game_state, depth, alpha, beta, maximizing_player, player):
    if depth == 0:
        return heuristic(game_state, player), None

    if player == 1 :
        target_player = game_state.player1
        opponent = 2
    else:
        target_player = game_state.player2
        opponent = 1

    if maximizing_player:
            best_val = float("-inf")
    else:
            best_val = float("inf")
    best_move = None
    moves = ["LEFT","STRAIGHT","RIGHT"]

    for move in moves:

        temp_state = copy.deepcopy(game_state)
        
        if (maximizing_player):
            # We're maximizing, move us around
            target = player
            if debug: print(f'{"    "*depth}(MinMax @ {depth}) Player {player} is facing {target_player.direction}, Checking {move} at depth {depth}!', flush=True)
        else:
            # We're moving THEM around to get us the smallest for US
            if debug: print(f'{"    "*depth}(MinMax @ {depth}) Opponent {opponent} is facing {target_player.direction}, Checking {move} at depth {depth}!', flush=True)
            target = opponent
        
        child = game_state.next_state(temp_state, move, player, target)
        eval, _ = minimax(child, depth - 1, alpha, beta, not maximizing_player, player)

        if maximizing_player:
            if debug:  print(f'{"    "*depth}(MinMax @ {depth}) MAXIMIZE START {eval}  {best_val}')
            if eval > best_val:
                if best_val is not None:
                    if debug:  print(f'{"    "*(depth + 1)}(MinMax @ {depth}) Move {move} {eval} was bigger than {best_move} {best_val}')
                best_val = eval
                best_move = move
            alpha = max(alpha, eval)
        else:
            if debug:  print(f'{"    "*depth}(MinMax @ {depth}) minimizing...')
            if eval < best_val:
                if best_val is not None:
                    if debug:  print(f'{"    "*depth}(MinMax @ {depth}) {eval} was smaller than{best_move} {best_val}')
                best_val = eval
                best_move = move
            beta = min(beta, eval)
        if debug: print(f'{"    "*depth}Done with {move}')
        # if beta <= alpha:
        #     print(f'!!!PRUNED!!! {beta} <= {alpha}  ')
        #     break
    if debug:  
        print(f'{"    "*depth}(MinMax @ {depth}) **EndMinMax Player: {target} - {best_move} at {best_val}')
    return best_val, best_move

def iterative_deepening(game_state, max_depth, player):
    best_move = None
    status = game_state.get_status()
    for depth in range(1, max_depth + 1):
        eval, move = minimax(
            game_state, max_depth, float("-inf"), float("inf"), True, player
    )
    if move:
        best_move = move
    return eval, best_move


def decide_move(game_state, max_depth, player):
    if player == 1:
        target = game_state.player1
    else:
        target = game_state.player2
    temp_state = copy.deepcopy(game_state)
    if debug:  print(f'#### Player {player} @ ({target.x},{target.y}) facing {target.direction} is thinking.... ####')
    eval, best_move = iterative_deepening(temp_state, max_depth, player)
    if debug:  print(f'==== Player {player} @ ({target.x},{target.y}) facing {target.direction} finally picked {best_move} ({eval}) ====\n')
    #stateUtils.gamestateSnapshot(game_state.MAP_SIZE , game_state)
    return best_move
