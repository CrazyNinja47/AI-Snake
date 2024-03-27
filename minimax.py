from snakebattle import GameState

# Main issue with minimax is we need to get future states, to be added to GameState


def heuristic(game_state):

    player1 = game_state.player1
    player2 = game_state.player2
    food_x, food_y = game_state.food
    winner = game_state.winner

    score = 0

    score += player2.length
    score -= player1.length

    player_distance = abs(player1.x - player2.x) + abs(player1.y - player2.y)

    # Reward for moving perpendicular to opponent
    if player2.direction in ["UP", "DOWN"] and player1.direction in ["LEFT", "RIGHT"]:
        score += 1
    elif player2.direction in ["LEFT", "RIGHT"] and player1.direction in ["UP", "DOWN"]:
        score += 1

    # Reward for being close to the opponent
    if score > 0:
        score /= player_distance

    # Reward heavily for moving towards food
    food_distance = abs(player2.x - food_x) + abs(player2.y - food_y)
    dis_score = 1 / (food_distance + 0.1)
    dis_score *= 5

    score += dis_score

    # Reward for winning or losing
    if winner == 1:
        score -= 1000
    elif winner == 2:
        score += 1000

    return score


def minimax(game_state, depth, alpha, beta, maximizing_player):
    if depth == 0 or game_state.is_terminal():
        return heuristic(game_state)

    if maximizing_player:
        max_eval = float("-inf")
        for child in game_state.get_children():
            eval = minimax(child, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float("inf")
        for child in game_state.get_children():
            eval = minimax(child, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def iterative_deepening(game_state, max_depth):

    best_move = None
    for depth in range(1, max_depth + 1):
        best_move = minimax(game_state, depth, float("-inf"), float("inf"), True)
    return best_move


def decide_move(game_state, max_depth):

    best_move = iterative_deepening(game_state, max_depth)
    return best_move
