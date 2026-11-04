from collections import deque

CAP_A, CAP_B, GOAL = 5, 3, 4


def next_states(state):
    """Return all valid next states and actions from a given state."""
    a, b = state
    result = [
        ((CAP_A, b), "Fill 5L jug"),
        ((a, CAP_B), "Fill 3L jug"),
        ((0, b), "Empty 5L jug"),
        ((a, 0), "Empty 3L jug"),
    ]

    amount = min(a, CAP_B - b)
    result.append(((a - amount, b + amount), "Pour 5L -> 3L"))

    amount = min(b, CAP_A - a)
    result.append(((a + amount, b - amount), "Pour 3L -> 5L"))

    return result


def bfs():
    queue = deque([((0, 0), [])])
    seen = {(0, 0)}

    while queue:
        state, path = queue.popleft()

        if GOAL in state:
            return path + [(state, "GOAL")]

        for next_state, action in next_states(state):
            if next_state not in seen:
                seen.add(next_state)
                queue.append((next_state, path + [(state, action)]))

    return None


def dfs():
    stack = [((0, 0), [])]
    seen = set()

    while stack:
        state, path = stack.pop()

        if state in seen:
            continue

        seen.add(state)

        if GOAL in state:
            return path + [(state, "GOAL")]

        for next_state, action in reversed(next_states(state)):
            if next_state not in seen:
                stack.append((next_state, path + [(state, action)]))

    return None


def show(name, sol):
    print(f"\n{name} SOLUTION")
    for i, (state, action) in enumerate(sol or []):
        print(f"{i:02d}. State={state} | {action}")


if __name__ == "__main__":
    print("EXPERIMENT 1A - WATER JUG | Goal: 4 litres")
    show("BFS", bfs())
    show("DFS", dfs())
