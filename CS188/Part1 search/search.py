# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


class node:
    def __init__(self, state, parent, action, cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.cost = cost

def getpath(node):
        path = []
        while node.parent != None:
            path.append(node.action)
            node = node.parent
        path.reverse()
        return path
def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
"""



    "*** YOUR CODE HERE ***"

    ##depthfirstsearchfunction

    openstack = util.Stack()
    root= node(problem.getStartState(), None, None, 0)
    openstack.push(root)
    closedlist = []
    while not openstack.isEmpty():
        current = openstack.pop()
        if problem.isGoalState(current.state):
            return getpath(current)
        elif current.state not in closedlist:
            closedlist.append(current.state)
            for successor in problem.getSuccessors(current.state):
                if successor[0] not in closedlist:
                    openstack.push(node(successor[0], current, successor[1], successor[2]))
    return []












def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    openQueue = util.Queue()
    root= node(problem.getStartState(), None, None, 0)
    openQueue.push(root)
    closedlist = []
    while not openQueue.isEmpty():
        current = openQueue.pop()
        if problem.isGoalState(current.state):
            return getpath(current)
        elif current.state not in closedlist:
            closedlist.append(current.state)
            for successor in problem.getSuccessors(current.state):
                if successor[0] not in closedlist:
                    openQueue.push(node(successor[0], current, successor[1], successor[2]))
    return []

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    openPQueue = util.PriorityQueue()
    root = node(problem.getStartState(), None, None, 0)
    openPQueue.push(root, root.cost)
    closedlist = []
    while not openPQueue.isEmpty():
        current = openPQueue.pop()

        if problem.isGoalState(current.state):

                return getpath(current)
        elif current.state not in closedlist:
            closedlist.append(current.state)
            for successor in problem.getSuccessors(current.state):
             if successor[0] not in closedlist:
                        openPQueue.update(node(successor[0], current, successor[1], current.cost+successor[2]), current.cost+successor[2])


    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    def priorityfunction(node):
        return node.cost + heuristic(node.state, problem)
    openPQueue = util.PriorityQueueWithFunction(priorityfunction)
    root = node(problem.getStartState(), None, None, 0)
    openPQueue.push(root)
    closedlist = []
    while not openPQueue.isEmpty():
        current = openPQueue.pop()
        if problem.isGoalState(current.state):

            return getpath(current)
        elif current.state not in closedlist:
            closedlist.append(current.state)
            for successor in problem.getSuccessors(current.state):
                if successor[0] not in closedlist:
                    openPQueue.push(node(successor[0], current, successor[1], current.cost + successor[2]))


    return []

    



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
