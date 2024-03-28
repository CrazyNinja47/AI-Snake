# Main issue with minimax is we need to get future states, to be added to GameState


def heuristic(game_state):

    player1 = game_state.player1
    player2 = game_state.player2
    food_x, food_y = game_state.food
    winner = game_state.winner

    score = 0

    score += player2.length * 50
    score -= player1.length * 50

    player_distance = abs(player1.x - player2.x) + abs(player1.y - player2.y)

    # Reward for moving perpendicular to opponent
    # if player2.direction in ["UP", "DOWN"] and player1.direction in ["LEFT", "RIGHT"]:
    #     score += 1
    # elif player2.direction in ["LEFT", "RIGHT"] and player1.direction in ["UP", "DOWN"]:
    #     score += 1

    aggro_score = 0
    # Reward for being close to the opponent as it gets longer
    if player_distance > 0:
        aggro_score += 5 / player_distance
        aggro_score *= player2.length * 10
        # If there is no food on the map, act much more aggressively - otherwise it will just go in circles
        if (food_x is None) or (food_y is None):
            aggro_score *= 50

    score += aggro_score

    food_distance = 0

    # Penalize for moving away from food
    if (food_x is not None) and (food_y is not None):
        food_distance = abs(player2.x - food_x) + abs(player2.y - food_y)
        dis_score = food_distance
    else:
        dis_score = 0

    score -= dis_score

    # print(f"[{player2.x}, {int(player2.y)}] [{food_x}, {food_y}], {food_distance}")

    if (food_x is not None) and (food_y is not None):
        if (int(player2.x) == int(food_x)) and (int(player2.y) == int(food_y)):
            print("Getting Food")
            score += 100

    # Reward for winning or losing
    if winner == 1:
        score -= 1000
    if winner == 2:
        score += 1000

    return score


def minimax(game_state, depth, alpha, beta, maximizing_player):
    if depth == 0 or game_state.is_terminal():
        return heuristic(game_state), None

    if maximizing_player:
        max_eval = float("-inf")
        best_move = None
        for child, move in game_state.get_children():
            eval, _ = minimax(child, depth - 1, alpha, beta, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float("inf")
        best_move = None
        for child, move in game_state.get_children():
            eval, _ = minimax(child, depth - 1, alpha, beta, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move


def iterative_deepening(game_state, max_depth):
    best_move = None
    for depth in range(1, max_depth + 1):
        eval, move = minimax(game_state, depth, float("-inf"), float("inf"), True)
        if move:
            best_move = move
    return eval, best_move


def decide_move(game_state, max_depth):
    eval, best_move = iterative_deepening(game_state, max_depth)
    # print(eval, best_move)
    return best_move
