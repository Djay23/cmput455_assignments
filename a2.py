# CMPUT 455 Assignment 2 starter code
# This is a sample solution for assignment 1.
# Implement the new commands to complete the assignment
# Full assignment specification and game rules on Canvas

import ast
import random
import time
from sys import stderr
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Assignment restrictions (see assignment page)
# ---------------------------------------------------------------------------
MIN_HEAPS = 1
MAX_HEAPS = 20
MIN_TOKENS = 1
MAX_TOKENS = 20
MIN_VALUE = 1
MAX_VALUE = 20
KOMI_LIMIT = 100  # -100 < komi < 100, and komi must be an integer + 0.5

# The time limit, in seconds, for a single solve or genmove command
MIN_TIMELIMIT = 1
MAX_TIMELIMIT = 100
DEFAULT_TIMELIMIT = 1

BLACK = 'b'
WHITE = 'w'
COLORS = (BLACK, WHITE)

Token = Tuple[str, int]
Heap = List[Token]
Heaps = List[Heap]


def not_yet() -> bool:
    raise NotImplementedError("Command not implemented.")
    return False


def print_error(error: str) -> None:
    print(error, file=stderr)


def opponent(color: str) -> str:
    return WHITE if color == BLACK else BLACK


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def is_int(obj: Any) -> bool:
    # bool is a subclass of int; True/False are not valid values
    return isinstance(obj, int) and not isinstance(obj, bool)


def is_token(obj: Any) -> bool:
    return (isinstance(obj, tuple)
            and len(obj) == 2
            and obj[0] in COLORS
            and is_int(obj[1])
            and MIN_VALUE <= obj[1] <= MAX_VALUE)


def is_heap(obj: Any) -> bool:
    return (isinstance(obj, list)
            and MIN_TOKENS <= len(obj) <= MAX_TOKENS
            and all(is_token(token) for token in obj))


def is_heaps(obj: Any) -> bool:
    return (isinstance(obj, list)
            and MIN_HEAPS <= len(obj) <= MAX_HEAPS
            and all(is_heap(heap) for heap in obj))


def parse_komi(text: str) -> Optional[float]:
    try:
        komi = float(text)
    except ValueError:
        return None
    if not (-KOMI_LIMIT < komi < KOMI_LIMIT):
        return None
    if (komi - 0.5) != int(komi - 0.5):
        return None
    return komi


def parse_heaps(text: str) -> Optional[Heaps]:
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None
    if not is_heaps(parsed):
        return None
    return parsed


def parse_heap_number(args: str) -> Optional[int]:
    parts = args.split()
    if len(parts) != 1:
        return None
    try:
        return int(parts[0])
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------
class HeapGo:
    def __init__(self, komi: float, heaps: Heaps) -> None:
        self.komi = komi
        self.heaps = heaps
        self.toplay = BLACK
        self.score = {BLACK: 0, WHITE: komi}

    def __str__(self) -> str:
        return "k {} {}".format(self.komi, self.heaps)

    def is_legal(self, heap_number: int) -> bool:
        return 0 <= heap_number < len(self.heaps) and len(self.heaps[heap_number]) > 0

    def legal_moves(self) -> List[int]:
        return [i for i in range(len(self.heaps)) if self.is_legal(i)]

    def game_over(self) -> bool:
        return len(self.legal_moves()) == 0

    def play(self, heap_number: int) -> bool:
        if not self.is_legal(heap_number):
            return False
        heap = self.heaps[heap_number]
        gained = 0
        # Rule 1: take all own-color tokens from the top
        while heap and heap[-1][0] == self.toplay:
            gained += heap.pop()[1]
        # Rule 2: then one opponent token, if any remain
        if heap:
            gained += heap.pop()[1]
        self.score[self.toplay] += gained
        self.toplay = opponent(self.toplay)
        return True

    def winner(self) -> Optional[str]:
        if not self.game_over():
            return None
        if self.score[BLACK] > self.score[WHITE]:
            return BLACK
        if self.score[WHITE] > self.score[BLACK]:
            return WHITE
        return None  # draw; impossible with komi = integer + 0.5

"""
Minimax search with transposition table
"""
# Compact Bit representation of a token: 0 - 20 = black and 32 - 52 = white to save search time
WHITE_BIT = 32                      # White token (in bits)
VALUE_MASK = WHITE_BIT - 1          # 31: keeps only the value part of a token
TABLE_LIMIT = 1_500_000             # max table entries to keeps memory under 1GB

IntHeaps = List[List[int]]          # top token of each heap is last
Move = Tuple[int, int, int]         # (points, heap index, new heap length)

# Store Transposition table with (player, score_diff, sorted heaps) to check if state has been checked, to reduce time
TranspositionTable = Dict[Tuple[int, float, bytes], bool]

def get_moves(heaps: IntHeaps, player: int) -> List[Move]:
    """
    Play one move per distinct heap with biggest gain first
    Returns a list of moves, each move is a tuple (points, heap index, new heap length)"""
    moves = {}
    for index, heap in enumerate(heaps):
        if not heap:
            continue

        # Find the new length of the heap after move 1 (Take own colors first then opponents colors)
        new_len, points = len(heap), 0
        while new_len and (heap[new_len - 1] >= WHITE_BIT) == player:  
            new_len -= 1
            points += heap[new_len] & VALUE_MASK

        if new_len:
            new_len -= 1
            points += heap[new_len] & VALUE_MASK

        # Store best move for canonical representation of heaps
        moves.setdefault(bytes(heap), (points, index, new_len))
    # Greedy method to find winning move faster by searching by largest gain first
    return sorted(moves.values(), reverse=True)

def search(heaps: IntHeaps, player: int, score_diff: float, token_value_left: int, deadline: float, table: TranspositionTable) -> Optional[bool]:
    """
    Search function returns True if the player to move plays perfectly and wins
    Returns False if the player to move plays perfectly and loses
    Returns None if the search times out before a result is found
    """

    # Exit game early without exploring moves if the winner is already clear or time has run out
    if score_diff - token_value_left > 0:
        return True
    if score_diff + token_value_left < 0:
        return False
    if time.monotonic() > deadline:
        return None

    # Store each state in  transposition table using a canonical representation of the heaps
    key = (player, score_diff, b"|".join(sorted(bytes(h) for h in heaps if h)))

    if key in table:
        return table[key]
    win = False

    # Simulates all possible moves for the current player and recursively searches
    for points, index, new_len in get_moves(heaps, player):
        heap = heaps[index]     # Select heap to alter
        tokens_removed = heap[new_len:]
        del heap[new_len:]      # Deletes tokens from the real heap to simulate a move

        opponent_wins = search(heaps, 1 - player, -(score_diff + points), token_value_left - points, deadline, table)

        # Mimic undo move and restore heap to previous state
        heap.extend(tokens_removed)

        # Return None if the search timed out at child node
        if opponent_wins is None:                         
            return None

        # Winning move was found if opposing color doesn't win
        if not opponent_wins:                           
            win = True
            break

    # Save game result while there is room in transposition table
    if len(table) < TABLE_LIMIT:
        table[key] = win
    return win

def minimax(game: HeapGo, deadline: float, table: Optional[TranspositionTable] = None) -> Optional[bool]:
    """
    Minimax algorithm for solving HeapGo

    Returns True if game.toplay wins with perfect play, False if it loses, None if
    time.monotonic() passes deadline first

    Transposition table stays the same within a solve to reduce time. This table is optional and can be created each time for the same solve
    """
    if table is None:
        table = {}
    # Encode the game into the compact form used by the search, minus empty heaps
    heaps = [[value + (WHITE_BIT if color == WHITE else 0) for color, value in heap] for heap in game.heaps if heap]

    token_value_left = sum(token & VALUE_MASK for heap in heaps for token in heap)
    score_diff = game.score[game.toplay] - game.score[opponent(game.toplay)]
    player = 1 if game.toplay == WHITE else 0

    return search(heaps, player, score_diff, token_value_left, deadline, table)

CommandMap = Dict[str, Callable[[str], bool]]

class CommandInterface:
    def __init__(self) -> None:
        self.game: Optional[HeapGo] = None
        self.timelimit = DEFAULT_TIMELIMIT
        # you can add your own initialisation here
        self.commands: CommandMap = {
            "help": self.cmd_help,
            "heapgo": self.cmd_heapgo,
            "show": self.cmd_show,
            "toplay": self.cmd_toplay,
            "play": self.cmd_play,
            "legal": self.cmd_legal,
            "genmove": self.cmd_genmove,
            "score": self.cmd_score,
            "winner": self.cmd_winner,
            "solve": self.cmd_solve,
            "timelimit": self.cmd_timelimit,
            }

#============================================================================
# Command implementations
#============================================================================
    def cmd_heapgo(self, args: str) -> bool:
        parts = args.split(maxsplit=1)
        if len(parts) != 2:
            print_error("usage: heapgo komi game")
            return False
        komi = parse_komi(parts[0])
        if komi is None:
            print_error("invalid komi: {}".format(parts[0]))
            return False
        heaps = parse_heaps(parts[1])
        if heaps is None:
            print_error("invalid game: {}".format(parts[1]))
            return False
        self.game = HeapGo(komi, heaps)
        return True

    def cmd_show(self, args: str) -> bool:
        if self.game is None:
            print_error("no game started")
            return False
        print(self.game)
        return True

    def cmd_toplay(self, args: str) -> bool:
        if self.game is None:
            print_error("no game started")
            return False
        if args not in COLORS:
            print_error("invalid color: {}".format(args))
            return False
        self.game.toplay = args
        return True

    def cmd_play(self, args: str) -> bool:
        if self.game is None:
            print_error("no game started")
            return False
        heap_number = parse_heap_number(args)
        if heap_number is None:
            print_error("invalid heap number: {}".format(args))
            return False
        if not self.game.play(heap_number):
            print_error("illegal move: {}".format(heap_number))
            return False
        return True

    def cmd_legal(self, args: str) -> bool:
        if self.game is None:
            print_error("no game started")
            return False
        heap_number = parse_heap_number(args)
        if heap_number is None:
            print_error("invalid heap number: {}".format(args))
            return False
        print("yes" if self.game.is_legal(heap_number) else "no")
        return True

    def cmd_genmove(self, args: str) -> bool:
        if self.game is None:
            print_error("no game started")
            return False
        moves = self.game.legal_moves()
        if not moves:
            print_error("no legal moves")
            return False
        # you need to replace this random move by a move from your solver
        move = random.choice(moves)
        self.game.play(move)
        print(move)
        return True

    def cmd_score(self, args: str) -> bool:
        if self.game is None:
            print_error("no game started")
            return False
        print("b {} w {}".format(self.game.score[BLACK], self.game.score[WHITE]))
        return True

    def cmd_winner(self, args: str) -> bool:
        if self.game is None:
            print_error("no game started")
            return False
        winner = self.game.winner()
        if winner is None:
            print_error("game not over" if not self.game.game_over() else "draw")
            return False
        print(winner)
        return True

#============================================================================
# You need to implement the following methods.
#============================================================================
    def cmd_solve(self, args: str) -> bool:
        return not_yet()

    def cmd_timelimit(self, args: str) -> bool:
        return not_yet()
#============================================================================
# End of functions requiring implementation
#============================================================================

#============================================================================
# The code below should not need modification
# Anyway, you may change or add to this code as you see fit
#============================================================================
    def cmd_help(self, ignore_args: str) -> bool:
        print("\nKnown commands:")
        for cmd in self.commands:
            print(cmd)
        return True

    def process_command(self, cmd_name: str, cmd_args: str) -> None:
        status = "= -1"
        cmd = self.commands.get(cmd_name)
        if cmd:
            try:
                if cmd(cmd_args):
                    status = "= 1"
            except Exception as e:
                print_error(f"Command {cmd_name} with arguments {cmd_args} failed with exception: {e}")
        else:
            print_error("Unknown command. Type 'help' for commands.")
        print(status)

    def main_loop(self) -> None:
        process_commands = True
        while process_commands:
            try:
                line = input()
            except EOFError:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            cmd_name = parts[0]
            if cmd_name == "exit":
                process_commands = False
                continue
            cmd_args = parts[1] if len(parts) > 1 else ""
            self.process_command(cmd_name, cmd_args)


if __name__ == "__main__":
    interface = CommandInterface()
    interface.main_loop()
