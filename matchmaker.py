import subprocess
import threading
import queue


games = 10
batchSize = 5

setup = ["--headless", "--p1=minimax", "--p2=minimax"]


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


def worker(game_queue, result_queue):
    while not game_queue.empty():
        try:
            game_id = game_queue.get_nowait()
            output = run_game("snakebattle.py", *setup)
            winner_line = parse_winner(output)
            print(f"Game #{game_id}: {winner_line}")
            result_queue.put(winner_line)
        finally:
            game_queue.task_done()


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
