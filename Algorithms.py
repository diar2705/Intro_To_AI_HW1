import numpy as np

from CampusEnv import CampusEnv
from typing import List, Tuple
import heapdict
from collections import deque


class Node:
    def __init__(
        self, state, parent=None, action=0, cost=0, terminated=False, h=0, f=0
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.terminated = terminated
        self.g = parent.g + cost if parent is not None else cost
        self.h = h
        self.f = f

    def __repr__(self) -> str:
        return f"{self.state}"


class Agent:
    def __int__(self):
        self.env = CampusEnv
        self.open = None
        self.close: set = None
        self.expanded: int = 0

    # expands the given node and returns its children
    def expand(self, node: Node):
        self.expanded += 1
        for action, (state, cost, termenated) in self.env.succ(node.state).items():
            if state != None:
                if state == node.state:
                    continue
                child = Node(
                    state, parent=node, action=action, cost=cost, terminated=termenated
                )
                yield child

    # returns the solution
    def solution(self, node: Node) -> Tuple[List[int], float, int]:
        total = 0
        actions = []

        while node.parent != None:
            total += node.cost
            actions.append(node.action)
            node = node.parent

        return list(reversed(actions)), total, self.expanded

    # initializes the search 
    def init_search(self, env: CampusEnv):
        self.env = env
        self.env.reset()
        self.expanded = 0
        self.close = set()


class DFSGAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    # the recrsive function of the DFS algorithm
    def dfs(self) -> Tuple[List[int], float, int]:
        node = self.open.pop()
        self.close.add(node.state)
        
        # we have the reached a goal state
        if self.env.is_final_state(node.state):
            return self.solution(node)
        
        for child in self.expand(node):
            # child isn't in close nor in open
            if child.state not in self.close and child.state not in [
                n.state for n in self.open
            ]:
                self.open.append(child)
                result = self.dfs()
                if result != None:
                    return result
                
        # there is no path from the start state to a goal state
        return None

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        self.init_search(env)

        self.open: deque = deque()
        node: Node = Node(env.get_initial_state())
        self.open.append(node)

        return self.dfs()


class UCSAgent(Agent):

    def __init__(self) -> None:
        super().__init__()

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        self.init_search(env)
        
        node: Node = Node(self.env.get_initial_state())
        self.open: heapdict = heapdict.heapdict()
        
        # the initial state is a goal state
        if self.env.is_final_state(node.state):
            return self.solution(node)

        self.open[node] = (node.g, node.state)

        # while open isn't empty
        while len(self.open) > 0:
            node, _ = self.open.popitem()
            self.close.add(node.state)
            
            # we have the reached a goal state
            if self.env.is_final_state(node.state):
                return self.solution(node)

            for child in self.expand(node):
                # the child isn't in close nor in open
                if child.state not in self.close and child.state not in [
                    n.state for n in self.open
                ]:
                    self.open[child] = (child.g, child.state)
                    
                # the child isn't in close but is in open
                elif child.state in [n.state for n in self.open]:
                    for n in self.open:
                        if n.state == child.state:
                            # the current path to the child has a lower cost than the previous path (to the same child)
                            if child.g < n.g:
                                del self.open[n]
                                self.open[child] = (child.g, child.state)

        # there is no path from the start state to a goal state
        return None


class WeightedAStarAgent(Agent):

    def __init__(self):
        super().__init__()

    # heuristic that calculates the Manhattan distance
    def h(self, state):
        curr_row, curr_col = self.env.to_row_col(state)
        goals = [self.env.to_row_col(goal) for goal in self.env.get_goal_states()]
        manhattan = [
            abs(curr_row - goal_row) + abs(curr_col - goal_col)
            for goal_row, goal_col in goals
        ]
        return min(manhattan + [100])

    def f(self, node: Node):
        return self.weight * node.h + (1 - self.weight) * node.g

    def search(self, env: CampusEnv, h_weight=0.5) -> Tuple[List[int], float, int]:
        self.weight = h_weight
        self.init_search(env)

        start: Node = Node(self.env.get_initial_state())
        self.open: heapdict = heapdict.heapdict()

        start.h = self.h(start.state)
        start.f = self.f(start)
        self.open[start] = (start.f, start.state)

        # while open isn't empty
        while len(self.open) > 0:
            node, _ = self.open.popitem()
            self.close.add(node)
            
            # we have the reached a goal state
            if self.env.is_final_state(node.state):
                return self.solution(node)

            for child in self.expand(node):
                child.h = self.h(child.state)
                child.f = self.f(child)

                # the child isn't in close nor in open
                if child.state not in [
                    n.state for n in self.close
                ] and child.state not in [n.state for n in self.open]:
                    self.open[child] = (child.f, child.state)

                # the child isn't in close but is in open
                elif child.state in [n.state for n in self.open]:
                    for n in self.open:
                        if n.state == child.state:
                            # the current path to the child has a lower cost than the previous path (to the same child)
                            if child.f < n.f:
                                del self.open[n]
                                self.open[child] = (child.f, child.state)
                                break
                
                # the child is in close but not in open
                else:
                    for n in self.close:
                        if n.state == child.state:
                            # the current path to the child has a lower cost than the previous path (to the same child)
                            if child.f < n.f:
                                self.close.remove(n)
                                self.open[child] = (child.f, child.state)
                                break
        
        # there is no path from the start state to a goal state
        return None


class AStarAgent(WeightedAStarAgent):

    def __init__(self):
        super().__init__()

    # calls the Weighted A* algorithm with the default weight value which is weight=0.5
    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        return super().search(env)
