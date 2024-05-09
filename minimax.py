# Main issue with minimax is we need to get future states, to be added to GameState
import copy as copy
import random as random
import logger
from collections import deque

LENGTH_WEIGHT = 10
AGGRO_WEIGHT = 15
FOOD_DIST_WEIGHT = 10
EATING_WEIGHT = 500
WIN_WEIGHT = 3000
LOSE_WEIGHT = 3000
DRAW_WEIGHT = 2000
EDGE_WEIGHT = 5
ENCLOSED_PENALTY = 100
# MAX_SQUARES_FLOODED = 75

debug = False


def heuristic(game_state, player, node):

    test = {"length": 0, "aggro": 0, "food": 0, "win": 0, "edge": 0, "eaten": 0}

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

    test.update(({"length": score}))

    player_distance = abs(player1.x - player2.x) + abs(player1.y - player2.y)

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
    test.update(({"aggro": aggro_score}))

    food_distance = 0

    # Penalize for moving away from food
    if (food_x is not None) and (food_y is not None):
        if player == 2 and not game_state.player2_full:
            food_distance = abs(player2.x - food_x) + abs(player2.y - food_y)
        elif not game_state.player1_full:
            food_distance = abs(player1.x - food_x) + abs(player1.y - food_y)
        dis_score = food_distance * FOOD_DIST_WEIGHT
    else:
        dis_score = 0

    test.update(({"food": -(dis_score)}))
    score -= dis_score

    # Reward for eating, not spinning around food!
    eaten_score = 0
    if player == 2:
        if game_state.player2_full:
            test.update(({"eaten": EATING_WEIGHT}))
            eaten_score = EATING_WEIGHT
        if game_state.player1_full:
            test.update(({"eaten": -EATING_WEIGHT}))
            eaten_score = -EATING_WEIGHT

    else:
        if game_state.player1_full:
            test.update(({"eaten": EATING_WEIGHT}))
            eaten_score = EATING_WEIGHT
        if game_state.player2_full:
            test.update(({"eaten": -EATING_WEIGHT}))
            eaten_score = -EATING_WEIGHT

    score += eaten_score

    # Reward for winning or losing
    if player == 2:
        if winner == 1:
            test.update(({"win": -(LOSE_WEIGHT)}))
            score -= LOSE_WEIGHT
        if winner == 2:
            score += WIN_WEIGHT
            test.update(({"win": WIN_WEIGHT}))
        if winner == 0:
            score -= DRAW_WEIGHT
            test.update(({"win": -(DRAW_WEIGHT)}))
    if player == 1:
        if winner == 2:
            score -= LOSE_WEIGHT
            test.update(({"win": -LOSE_WEIGHT}))
        if winner == 1:
            test.update(({"win": WIN_WEIGHT}))
            score += WIN_WEIGHT
        if winner == 0:
            score -= DRAW_WEIGHT
            test.update(({"win": -(DRAW_WEIGHT)}))

    if player == 1:
        playerID = player1
        enemyID = player2
    else:
        playerID = player2
        enemyID = player1

    if playerID.x < 2 or playerID.x >= (game_state.MAP_SIZE[0] - 2):
        score -= EDGE_WEIGHT
        test.update(({"edge": -(EDGE_WEIGHT)}))
    if playerID.y < 2 or playerID.y >= (game_state.MAP_SIZE[1] - 2):
        score -= EDGE_WEIGHT
        test.update(({"edge": -(EDGE_WEIGHT)}))
    if enemyID.x < 2 or enemyID.x >= (game_state.MAP_SIZE[0] - 2):
        score += EDGE_WEIGHT
        test.update(({"edge": (EDGE_WEIGHT)}))
    if enemyID.y < 2 or enemyID.y >= (game_state.MAP_SIZE[1] - 2):
        test.update(({"edge": (EDGE_WEIGHT)}))
        score += EDGE_WEIGHT

    if player == 1:
        if is_enclosed(game_state, player1.x, player1.y, player1.length):
            score -= ENCLOSED_PENALTY
    else:
        if is_enclosed(game_state, player2.x, player2.y, player2.length):
            score -= ENCLOSED_PENALTY

    if debug:
        print(
            f"#**# (Heuristic) Player {player} @ ({playerID.x},{playerID.y}) with {test} and total: {round(sum(test.values()),4)} #**#"
        )
    node.heuristic = test
    return score


def minimax(game_state, depth, alpha, beta, maximizing_player, player, node, logging):
    if depth == 0:
        return heuristic(game_state, player, node), None

    if player == 1:
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
    moves = ["STRAIGHT", "LEFT", "RIGHT"]
    # random.shuffle(moves)

    for move in moves:
        temp_state = copy.deepcopy(game_state)
        if maximizing_player:
            # We're maximizing, move us around
            target = player
            if debug:
                print(
                    f'{"    "*depth}(MinMax @ {depth}) Player {player} is facing {target_player.direction}, Checking {move} at depth {depth}!',
                    flush=True,
                )
        else:
            # We're moving THEM around to get us the smallest for US
            if debug:
                print(
                    f'{"    "*depth}(MinMax @ {depth}) Opponent {opponent} is facing {target_player.direction}, Checking {move} at depth {depth}!',
                    flush=True,
                )
            target = opponent

        child = temp_state.next_state(temp_state, move, player, target)
        if child.winner == opponent:

            eval = heuristic(child, player, node)
        elif child.winner == 0:
            heuristic(child, player, node)
            eval = heuristic(child, player, node)
        elif child.winner == player:
            heuristic(child, player, node)
            eval = heuristic(child, player, node)
        else:
            next_node = logger.MinMax_Node()
            if move == "LEFT":
                node.left_child = next_node
            if move == "STRAIGHT":
                node.center_child = next_node
            if move == "RIGHT":
                node.right_child = next_node
            eval, _ = minimax(
                child,
                depth - 1,
                alpha,
                beta,
                not maximizing_player,
                player,
                next_node,
                logging,
            )

        if maximizing_player:
            if debug:
                print(
                    f'{"    "*depth}(MinMax @ {depth}) MAXIMIZE START {eval}  {best_val}'
                )
            if eval > best_val:
                if best_val is not None:
                    if debug:
                        print(
                            f'{"    "*(depth + 1)}(MinMax @ {depth}) Move {move} {eval} was bigger than {best_move} {best_val}'
                        )
                best_val = eval
                best_move = move
            alpha = max(alpha, eval)
        else:
            if debug:
                print(f'{"    "*depth}(MinMax @ {depth}) minimizing...')
            if eval < best_val:
                if best_val is not None:
                    if debug:
                        print(
                            f'{"    "*depth}(MinMax @ {depth}) {eval} was smaller than{best_move} {best_val}'
                        )
                best_val = eval
                best_move = move
            beta = min(beta, eval)
        if debug:
            print(f'{"    "*depth}Done with {move}')
        if beta <= alpha:
            if debug:
                print(f"!!!PRUNED!!! {beta} <= {alpha}  ")
            break
    if debug:
        print(
            f'{"    "*depth}(MinMax @ {depth}) **EndMinMax Player: {target} - {best_move} at {best_val}'
        )
    node.choice = best_move
    if node.left_child and best_move == "LEFT":
        node.heuristic = node.left_child.heuristic
    if node.center_child and best_move == "STRAIGHT":
        node.heuristic = node.center_child.heuristic
    if node.right_child and best_move == "RIGHT":
        node.heuristic = node.right_child.heuristic
    return best_val, best_move


def iterative_deepening(game_state, max_depth, player, player_root_node, logging):
    best_move = None
    status = game_state.get_status()
    # Disabled for speed purposes
    # for depth in range(1, max_depth + 1):
    eval, move = minimax(
        game_state,
        max_depth,
        float("-inf"),
        float("inf"),
        True,
        player,
        player_root_node,
        logging,
    )
    if move:
        best_move = move
    return eval, best_move


def decide_move(game_state, max_depth, player, step_logger, logging):
    node = logger.MinMax_Node()
    if player == 1:
        target = game_state.player1
        step_logger.p1_tree = node
    else:
        target = game_state.player2
        step_logger.p2_tree = node
    temp_state = copy.deepcopy(game_state)
    if debug:
        print(
            f"#### Player {player} @ ({target.x},{target.y}) facing {target.direction} is thinking.... ####"
        )
    eval, best_move = iterative_deepening(temp_state, max_depth, player, node, logging)
    if debug:
        print(
            f"==== Player {player} @ ({target.x},{target.y}) facing {target.direction} finally picked {best_move} ({eval}) ====\n"
        )
    # stateUtils.gamestateSnapshot(game_state.MAP_SIZE , game_state)
    return best_move


def is_enclosed(game_state, start_x, start_y, length):
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    visited = set()
    queue = deque([(start_x, start_y)])

    count = 0

    while queue:
        x, y = queue.popleft()
        if (x, y) in visited:
            continue
        visited.add((x, y))
        count += 1

        if count > (length * 1.5):
            return False

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            sqr = game_state.get_square(nx, ny)
            if sqr != 1:
                queue.append((nx, ny))

    return True
