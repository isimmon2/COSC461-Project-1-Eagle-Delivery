from collections import deque
import heapq
from itertools import count


class SearchResult:
    def __init__(self, path, expanded):
        self.path = path
        self.expanded = expanded


def reconstruct_path(parents, goal):
    path = []
    state = goal
    while state is not None:
        path.append(state)
        state = parents[state]
    path.reverse()
    return path


def bfs(campus, start, goal):
    frontier = deque([start])
    parents = {start: None}
    expanded = 0
    while frontier:
        state = frontier.popleft()
        if state == goal:
            return SearchResult(reconstruct_path(parents, goal), expanded)
        expanded += 1
        for action, successor in campus.neighbors(state):
            if successor not in parents:
                parents[successor] = state
                frontier.append(successor)
    return SearchResult(None, expanded)


def manhattan(state, goal):
    # Add the absolute row difference and the absolute column difference.
    return abs(state[0]-goal[0])+abs(state[1]-goal[1])


def astar(campus, start, goal):
    serial = 0
    frontier = [(manhattan(start, goal), serial,0, start)]
    heapq.heapify(frontier)

    best_g = {start: 0}
    parents = {start: None}
    expanded = 0

    while frontier:
        f, _, g, state = heapq.heappop(frontier)
        if g != best_g[state]:
            continue
        # Goal test
        if state == goal:
            return SearchResult(reconstruct_path(parents, goal), expanded)

        expanded += 1

        for action, successor in campus.neighbors(state):
            new_g = best_g[state] + 1

            if new_g < best_g.get(successor, float("inf")):
                best_g[successor] = new_g
                parents[successor] = state

                new_f = new_g + manhattan(successor, goal)

                serial += 1
                heapq.heappush(frontier,(new_f, serial,new_g, successor))
    return SearchResult(None, expanded)
def epsilon_greedy(q_values, state, actions, epsilon, rng):
    if rng.random() < epsilon:
        return rng.choice(actions)
    q_table = []

    for action in actions:
        q = q_values.get((state, action), 0.0)
        q_table.append(q)

    max_q = max(q_table)

    best_actions = []
    for i in range(len(actions)):
        if q_table[i] == max_q:
            best_actions.append(actions[i])

    return rng.choice(best_actions)

def q_learning_update(q_values, state, action, reward, next_state, actions, alpha, gamma, terminal):
    if terminal:
        y= reward

    else:
        max_q = 0.0

        for a in actions:
            q = q_values.get((next_state, a), 0.0)

            if q > max_q:
                max_q = q
        y = reward + gamma * max_q
    old_q = q_values.get((state, action), 0)

    q_values[(state, action)] = ( old_q + alpha * (y - old_q))