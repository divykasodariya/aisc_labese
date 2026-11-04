
import heapq

# Final goal configuration
GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)


# Calculate Manhattan Distance
def manhattan(state):

    distance = 0

    for i in range(9):

        tile = state[i]

        # Ignore the empty space
        if tile == 0:
            continue

        # Find the tile's correct position
        j = GOAL.index(tile)

        # Current row and column
        row = i // 3
        col = i % 3

        # Goal row and column
        goal_row = j // 3
        goal_col = j % 3

        # Add Manhattan distance
        distance += abs(row - goal_row) + abs(col - goal_col)

    return distance


# Generate all possible next states
def neighbors(state):

    result = []

    # Find position of empty space
    zero = state.index(0)

    row = zero // 3
    col = zero % 3

    # Possible moves: UP, DOWN, LEFT, RIGHT
    moves = [
        (-1, 0, "UP"),
        (1, 0, "DOWN"),
        (0, -1, "LEFT"),
        (0, 1, "RIGHT")
    ]

    for dr, dc, name in moves:

        new_row = row + dr
        new_col = col + dc

        # Check whether the new position is valid
        if 0 <= new_row < 3 and 0 <= new_col < 3:

            new_pos = new_row * 3 + new_col

            # Convert tuple to list so we can swap
            new_state = list(state)

            # Swap empty space with adjacent tile
            new_state[zero], new_state[new_pos] = new_state[new_pos], new_state[zero]

            # Store new state and move
            result.append((tuple(new_state), name))

    return result


# A* Search Algorithm
def solve(start):

    # Priority queue: (f value, counter, state)
    heap = [(manhattan(start), 0, start)]

    # Store actual cost to reach each state
    g = {start: 0}

    # Store parent state and action
    parent = {start: None}
    action = {}

    count = 0

    while heap:

        # Get state with lowest f value
        f, _, current = heapq.heappop(heap)

        # Check if goal is reached
        if current == GOAL:

            path = []

            # Trace back from goal to start
            while current != start:
                path.append((current, action[current]))
                current = parent[current]

            # Add starting state
            path.append((start, "START"))

            # Reverse to get start-to-goal path
            path.reverse()

            return path

        # Check all neighboring states
        for nxt, move in neighbors(current):

            # Cost of reaching the next state
            new_cost = g[current] + 1

            # Check if this is a new or cheaper path
            if nxt not in g or new_cost < g[nxt]:

                g[nxt] = new_cost

                # Store parent and move
                parent[nxt] = current
                action[nxt] = move

                count += 1

                # Calculate f = g + h
                priority = new_cost + manhattan(nxt)

                # Add to priority queue
                heapq.heappush(heap, (priority, count, nxt))

    return None


# Display puzzle state
def show(state):

    print(state[:3])
    print(state[3:6])
    print(state[6:])


if __name__ == "__main__":

    print("EXPERIMENT 1B - A* 8 PUZZLE")

    # Starting configuration
    start = (1, 2, 3, 5, 0, 6, 4, 7, 8)

    # Solve puzzle
    solution = solve(start)

    if solution is None:
        print("No solution found")

    else:
        for i, (state, move) in enumerate(solution):

            print("\nStep", i, "|", move)
            show(state)
