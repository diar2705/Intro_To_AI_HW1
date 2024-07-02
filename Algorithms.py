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
                                
                            

class WeightedAStarAgent():
    
    def __init__(self):
        raise NotImplementedError

    def search(self, env: CampusEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError   



class AStarAgent():
    
    def __init__(self):
        raise NotImplementedError

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError 

