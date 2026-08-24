# agent.py
import random
# ===== ADDED (Lab 03 - Step 1.2: imports needed for BFS/DFS/UCS) =====
from collections import deque
import heapq
import itertools
# ===== END ADDED =====
# ===== ADDED (Lab 04 - Step 1.1: needed for euclidean_distance) =====
import math
# ===== END ADDED =====


class GreedyGridAgent:
    """A simple agent that tries to move around systematically to clear the grid."""

    def __init__(self):
        self.actions_pool = ['Up', 'Down', 'Left', 'Right']

    def sense_and_act(self, percept: dict) -> str:
        pos = percept.get('agent_pos')
        return random.choice(self.actions_pool)


# ===== ADDED (Lab 02 - Step 1.2: Simple Reflex Agent) =====
class SimpleReflexAgent:
    """
    A Simple Reflex Agent

    Selects an action using ONLY the current percept - strictly IF-THEN
    Condition-Action rules. No __init__ memory, no history of any kind.
    This is exactly what makes it fail in a partially observable
    environment: it cannot tell that it has been in this exact situation
    before, so it will repeat the same mistake forever.
    """

    def sense_and_act(self, percept: dict) -> str:
        # --- Condition-Action Rules ---
        # IF food_here THEN suck (eat the food on the current cell)
        if percept.get('food_here'):
            return 'Suck'

        # IF wall_ahead THEN turn_left (attempt to move Left instead)
        if percept.get('wall_ahead'):
            return 'Left'

        # ELSE move_forward (default: keep moving Up)
        return 'Up'
# ===== END ADDED (Step 1.2) =====


# ===== ADDED (Lab 02 - Step 1.3: Model-Based Agent) =====
class ModelBasedAgent:
    """
    A Model-Based Reflex Agent (Lecture 03).

    Unlike SimpleReflexAgent, this agent maintains an internal memory state
    across calls, so it can recognize when it is repeating itself and break
    out of a loop that a Simple Reflex Agent would be stuck in forever.
    """

    def __init__(self):
        # --- Internal memory state ---
        self.percept_history = []          # every percept ever received
        self.action_history = []           # every action ever taken
        self.visited_cells = set()         # placeholder for spatial memory
        # Maps a percept "signature" -> the last action tried in response
        # to it. This is what lets the agent notice "I already tried Left
        # here and it didn't work, so this time I'll try something else."
        self.percept_action_memory = {}

    def _signature(self, percept: dict) -> tuple:
        """Turns a percept dict into a hashable signature so it can be used
        as a memory key."""
        return (percept.get('wall_ahead'), percept.get('food_here'))

    def sense_and_act(self, percept: dict) -> str:
        # --- Step 1: Update internal state (Transition Model + Sensor Model) ---
        # Transition Model: record what the world just showed us.
        self.percept_history.append(dict(percept))
        signature = self._signature(percept)

        # --- Step 2: Condition-Action rules that consult memory ---
        if percept.get('food_here'):
            action = 'Suck'

        elif percept.get('wall_ahead'):
            # Sensor Model: check what action we tried LAST TIME we were in
            # this exact situation (same percept signature).
            previous_response = self.percept_action_memory.get(signature)

            if previous_response == 'Left':
                # We already tried turning left from here and hit a wall
                # again -> memory tells us to try the opposite direction.
                action = 'Right'
            elif previous_response == 'Right':
                action = 'Down'
            else:
                # First time seeing this wall -> default reflex response.
                action = 'Left'
        else:
            action = 'Up'

        # Record what we just decided, keyed by the situation we saw,
        # so next time we recognize the repeat and can choose differently.
        self.percept_action_memory[signature] = action
        self.action_history.append(action)
        return action
# ===== END ADDED (Step 1.3) =====


# ===== ADDED (Lab 03 - Step 1.2 & 1.3: Search / Goal-Based Agent) =====
class SearchAgent:
    """
    A Goal-Based / Planning Agent (Lecture 04/05).

    Unlike the reflex agents above, this agent does NOT react turn-by-turn.
    Instead, whenever it has no plan, it forms a COMPLETE offline plan (a
    full sequence of actions) by running a search algorithm against the
    environment's exposed state space, then executes that plan one action
    at a time until it needs to re-plan.
    """

    # Movement directions shared by all search algorithms.
    _MOVES = [
        ('Up', (0, 1)),
        ('Down', (0, -1)),
        ('Left', (-1, 0)),
        ('Right', (1, 0)),
    ]

    def __init__(self):
        self.plan = []
        # ===== MODIFIED (Lab 04) ===== 'BFS', 'DFS', 'UCS', or 'AStar'
        self.active_algo = 'AStar'
        # ===== END MODIFIED =====

    # ---------- Step 1.2 (Lab 03): Uninformed Search Algorithms ----------

    def bfs_search(self, start_pos, goal_pos, walls, grid_size) -> list:
        """Breadth-First Search - FIFO frontier (deque.popleft()). Explores
        the shallowest nodes first, guaranteeing the shortest path."""
        width, height = grid_size
        wall_set = set(tuple(w) for w in walls)
        start = tuple(start_pos)
        goal = tuple(goal_pos)

        if start == goal:
            return []

        frontier = deque([(start, [])])
        reached = {start}  # ===== reached set: converts Tree Search -> Graph Search =====

        while frontier:
            pos, path = frontier.popleft()
            for action, (dx, dy) in self._MOVES:
                npos = (pos[0] + dx, pos[1] + dy)
                if not (0 <= npos[0] < width and 0 <= npos[1] < height):
                    continue
                if npos in wall_set or npos in reached:
                    continue
                new_path = path + [action]
                if npos == goal:
                    return new_path
                reached.add(npos)
                frontier.append((npos, new_path))

        return None

    def dfs_search(self, start_pos, goal_pos, walls, grid_size) -> list:
        """Depth-First Search - LIFO stack (list.pop()). Explores the
        deepest nodes first; finds *a* path, not necessarily the shortest."""
        width, height = grid_size
        wall_set = set(tuple(w) for w in walls)
        start = tuple(start_pos)
        goal = tuple(goal_pos)

        if start == goal:
            return []

        stack = [(start, [])]
        reached = {start}  # ===== reached set: prevents infinite loops in DFS =====

        while stack:
            pos, path = stack.pop()
            for action, (dx, dy) in self._MOVES:
                npos = (pos[0] + dx, pos[1] + dy)
                if not (0 <= npos[0] < width and 0 <= npos[1] < height):
                    continue
                if npos in wall_set or npos in reached:
                    continue
                new_path = path + [action]
                if npos == goal:
                    return new_path
                reached.add(npos)
                stack.append((npos, new_path))

        return None

    def ucs_search(self, start_pos, goal_pos, walls, grid_size) -> list:
        """Uniform-Cost Search - priority queue (heapq.heappop()) ordered
        by total path cost g(n). Every step costs 1 here, so UCS behaves
        like BFS, but generalizes to variable step costs."""
        width, height = grid_size
        wall_set = set(tuple(w) for w in walls)
        start = tuple(start_pos)
        goal = tuple(goal_pos)

        if start == goal:
            return []

        counter = itertools.count()  # tie-breaker so heapq never compares lists
        frontier = [(0, next(counter), start, [])]
        best_cost = {start: 0}

        while frontier:
            cost, _, pos, path = heapq.heappop(frontier)
            if cost > best_cost.get(pos, float('inf')):
                continue  # stale entry, a cheaper path to pos was already found
            if pos == goal:
                return path

            for action, (dx, dy) in self._MOVES:
                npos = (pos[0] + dx, pos[1] + dy)
                if not (0 <= npos[0] < width and 0 <= npos[1] < height):
                    continue
                if npos in wall_set:
                    continue
                new_cost = cost + 1  # step cost = 1
                if npos not in best_cost or new_cost < best_cost[npos]:
                    best_cost[npos] = new_cost
                    heapq.heappush(frontier, (new_cost, next(counter), npos, path + [action]))

        return None

    # ---------- Step 1.1 (Lab 04): Heuristic Functions ----------

    def manhattan_distance(self, pos, goal) -> int:
        """h(n) = |x1 - x2| + |y1 - y2|
        Admissible for 4-way (Up/Down/Left/Right) grid movement: it never
        overestimates the true number of steps needed, since every step
        changes exactly one coordinate by 1."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def euclidean_distance(self, pos, goal) -> float:
        """h(n) = sqrt((x1 - x2)^2 + (y1 - y2)^2)
        Straight-line distance. Still admissible here (it's always <= the
        true grid-step cost), but it underestimates more loosely than
        Manhattan distance on a 4-way grid, since diagonal movement isn't
        actually allowed."""
        return math.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

    # ---------- Step 1.2 (Lab 04): A* Search ----------

    def astar_search(self, start_pos, goal_pos, walls, grid_size, heuristic_type='manhattan') -> list:
        """A* Search - priority queue ordered by f(n) = g(n) + h(n).
        g(n) = path cost so far, h(n) = estimated cost to goal (heuristic).
        Combines UCS's guarantee of finding the cheapest path with a
        heuristic that focuses exploration toward the goal, so far fewer
        nodes are expanded than plain UCS/BFS."""
        width, height = grid_size
        wall_set = set(tuple(w) for w in walls)
        start = tuple(start_pos)
        goal = tuple(goal_pos)

        heuristic_fn = (self.manhattan_distance if heuristic_type == 'manhattan'
                        else self.euclidean_distance)

        if start == goal:
            return []

        g_start = 0
        h_start = heuristic_fn(start, goal)
        f_start = g_start + h_start

        # Frontier tuple format: (f_cost, g_cost, current_pos, path_taken)
        frontier = [(f_start, g_start, start, [])]
        reached_states = set()

        while frontier:
            f_cost, g_cost, current_pos, path_taken = heapq.heappop(frontier)

            if current_pos == goal:
                return path_taken

            if current_pos in reached_states:
                continue
            reached_states.add(current_pos)

            for action, (dx, dy) in self._MOVES:
                npos = (current_pos[0] + dx, current_pos[1] + dy)
                if not (0 <= npos[0] < width and 0 <= npos[1] < height):
                    continue
                if npos in wall_set or npos in reached_states:
                    continue

                g_new = g_cost + 1
                h_new = heuristic_fn(npos, goal)
                f_new = g_new + h_new
                heapq.heappush(frontier, (f_new, g_new, npos, path_taken + [action]))

        return None

    # ---------- Step 1.3: Executing the Offline Plan ----------

    def _closest_food(self, agent_pos, all_food):
        """Picks the food pellet with the smallest Manhattan distance from
        the agent's current position, to give the search a concrete goal."""
        if not all_food:
            return None
        ax, ay = agent_pos
        return min(all_food, key=lambda f: abs(f[0] - ax) + abs(f[1] - ay))

    def sense_and_act(self, percept: dict) -> str:
        # ===== ADDED (Lab 04 - Step 1.3) =====
        # Extract global state, including remaining_food, to decide whether
        # there's anything left worth planning a route toward.
        if percept.get('remaining_food', 1) == 0:
            return 'Up'
        # ===== END ADDED =====

        if not self.plan:
            agent_pos = tuple(percept['agent_pos'])
            all_food = [tuple(f) for f in percept.get('all_food', [])]
            walls = percept.get('walls', [])
            grid_size = percept.get('grid_size')

            goal = self._closest_food(agent_pos, all_food)
            if goal is None:
                # No food left to plan toward.
                return 'Up'

            if self.active_algo == 'BFS':
                path = self.bfs_search(agent_pos, goal, walls, grid_size)
            elif self.active_algo == 'DFS':
                path = self.dfs_search(agent_pos, goal, walls, grid_size)
            elif self.active_algo == 'UCS':
                path = self.ucs_search(agent_pos, goal, walls, grid_size)
            # ===== ADDED (Lab 04 - Step 1.3) =====
            elif self.active_algo == 'AStar':
                path = self.astar_search(agent_pos, goal, walls, grid_size)
            # ===== END ADDED =====
            else:
                path = None

            self.plan = path if path else []

            if not self.plan:
                # Goal unreachable - fall back to a single default move
                # rather than crashing.
                return 'Up'

        return self.plan.pop(0)
# ===== END ADDED (Lab 03 / Lab 04) =====