import numpy as np

from CampusEnv import CampusEnv
from typing import List, Tuple
import heapdict
from collections import deque

class Node():
    def __init__(self, state, parent=None, action=0, cost=0, terminated=False):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost
        self.terminated = terminated
        self.g = parent.g + cost if parent is not None else cost

    def __repr__(self) -> str:
        return f"{self.state}"
class Agent():
    def __int__(self):
        self.env = CampusEnv
        self.open = None
        self.close: set = None
        self.expanded: int=0
    def expand(self, node:Node) -> List[Node]:
        self.expanded += 1
        for action, (state, cost, termenated) in self.env.succ(node.state).items():
            if state != None:
                child = Node(state, parent=node, action=action, cost=cost, terminated=termenated)
                yield child

    def solution(self, node: Node) -> Tuple[List[int], int, int]:
        total = 0
        actions = []

        while node.parent != None:
            total += node.cost
            actions.append(node.action)
            node = node.parent

        return list(reversed(actions)), total, self.expanded

    def init_search(self, env: CampusEnv):
        self.env = env
        self.env.reset()
        self.expanded = 0
        self.close = set()

class DFSGAgent(Agent):
    def __init__(self) -> None:
        super().__init__()

    def dfs(self)->Tuple[List[int], int, int]:
        node = self.open.pop()
        self.close.add(node.state)

        if self.env.is_final_state(node.state):
            return self.solution(node)
        for child in self.expand(node):
            if child.state not in self.close and child.state not in [n.state for n in self.open]:
                self.open.append(child)
                result = self.dfs()
                if result != None :
                    return result
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

        if self.env.is_final_state(node.state):
            return self.solution(node)

        self.open[node] = (node.g, node.state)

        while len(self.open) > 0:
            node, _ = self.open.popitem()
            self.close.add(node.state)

            if self.env.is_final_state(node.state):
                return self.solution(node)
            
            for child in self.expand(node):
                if child.state not in self.close and child.state not in [n.state for n in self.open]:
                    self.open[child] = (child.g,child.state)
                elif child.state in [n.state for n in self.open]:
                    for n in self.open:
                        if n.state == child.state:
                            if child.g < n.g:
                                del self.open[n]
                                self.open[child] = (child.g,child.state)

        return None
                                
                            
class WNode(Node):
    def __int__(self, state, parent=None, action=0, cost=0, terminated=False, h=0, f=0):
        super().__init__(state, parent=parent, action=action, cost=cost, terminated=terminated)
        self.h = h
        self.f = f



class WeightedAStarAgent(Agent):

    def __init__(self):
        super().__init__()

    def h(self, state):
        curr_row, curr_col = self.env.to_row_col(state)
        goal_cor = [self.env.to_row_col(state) for state in self.env.get_goal_states()]
        manhattan = [abs(curr_row - goal_row) + abs(curr_col - goal_col) for goal_row, goal_col in goal_cor]
        return min(manhattan + [100])

    def f(self, node : WNode):
        if self.weight == 1:
            return node
        return self.weight * node.h + (1 - self.weight) * node.g

    def search(self, env: CampusEnv, h_weight=0.5) -> Tuple[List[int], float, int]:
        self.weight = h_weight
        self.init_search(env)

        node: Node = Node(self.env.get_initial_state())

        self.open: heapdict = heapdict.heapdict()

        if self.env.is_final_state(node.state):
            return self.solution(node)

        self.open[node] = (node.g, node.state)

        while len(self.open) > 0:
            node, _ = self.open.popitem()
            self.close.add(node)

            if self.env.is_final_state(node.state):
                return self.solution(node)

            for child in self.expand(node):
                if child not in self.close and child.state not in [n.state for n in self.open]:
                    self.open[child] = (child.g, child.state)
                elif child.state in [n.state for n in self.open]:
                    for n in self.open:
                        if n.state == child.state:
                            if child.g < n.g:
                                del self.open[n]
                                self.open[child] = (child.g, child.state)

                else:
                    for n in self.close:
                        if n.state == child.state:
                            if child.g < n.g:
                                self.close.remove(n)
                                self.open[child] = (child.g, child.state)

        return None




class AStarAgent(WeightedAStarAgent):
    
    def __init__(self):
        super().__init__()

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        return super().search(env)



def main():
    MAPS = {
        "4x4": ["SFFF",
                "FHFH",
                "FFFH",
                "HFFG"],
        "8x8": ["SFFFFFFF",
                "FFFFFTAL",
                "TFFHFFTF",
                "FPFFFHTF",
                "FAFHFPFF",
                "FHHFFFHF",
                "FHTFHFTL",
                "FLFHFFFG"],
    }
    env = CampusEnv(MAPS["8x8"])
    state = env.reset()
    print('Initial state:', state)
    print('Goal states:', env.get_goal_states())

    WAstar_agent = WeightedAStarAgent()
    actions, total_cost, expanded = WAstar_agent.search(env, h_weight=0.5)

    print(f"Actions: {actions}")
    print(f"Total cost: {total_cost}")
    print(f"Expanded nodes: {expanded}")

if __name__ == "__main__":
    main()