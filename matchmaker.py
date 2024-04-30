import subprocess
import threading
import queue
import math

games = 100
batchSize = 3
tiled = False

setup = ["--fps=15", "--p1=astar", "--p2=minimax"]

if not tiled:
    setup.append("--headless")


def run_game(script_path, *args):
    command = ["python", script_path] + list(args)
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        return e.stderr


def parse_winner(output):
    lines = output.split("\n")
    for line in lines:
        if "won" in line:
            return line.strip()
    return "No winner found"


size = math.ceil(math.sqrt(batchSize))


def worker(game_queue, result_queue):
    while not game_queue.empty():
        try:
            game_id = game_queue.get_nowait()
            if tiled:
                pos_x, pos_y = calculate_position(game_id, 300, 300, size, size)
                output = run_game(
                    "snakebattle.py", "--pos_x", str(pos_x), "--pos_y", str(pos_y)
                )
            else:
                output = run_game("snakebattle.py", *setup)
            winner_line = parse_winner(output)
            print(f"Game #{game_id}: {winner_line}")
            result_queue.put(winner_line)
        finally:
            game_queue.task_done()


def calculate_position(index, width, height, rows, columns):
    row = index // columns
    col = index % columns
    pos_x = col * (width + 10) + 300
    pos_y = row * (height + 10) + 100
    return pos_x, pos_y


def main(games, batchSize):
    game_queue = queue.Queue()
    result_queue = queue.Queue()

    for i in range(games):
        game_queue.put(i)

    threads = []
    for _ in range(min(batchSize, games)):
        t = threading.Thread(target=worker, args=(game_queue, result_queue))
        t.start()
        threads.append(t)

    game_queue.join()

    player_1_wins = 0
    player_2_wins = 0

    while not result_queue.empty():
        winner_line = result_queue.get()
        if "Player 1 won" in winner_line:
            player_1_wins += 1
        elif "Player 2 won" in winner_line:
            player_2_wins += 1

    print("Player 1 Wins:", player_1_wins)
    print("Player 2 Wins:", player_2_wins)

    for t in threads:
        t.join()


if __name__ == "__main__":
    main(games, batchSize)
