# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util, math

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        curPos = currentGameState.getPacmanPosition()
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        curFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]

        score = 0
        for i in range(len(ghostPositions)):
            ghostNewDist = manhattanDistance(newPos, ghostPositions[i])
            if newScaredTimes[i] > ghostNewDist * 1.5 and newScaredTimes[i] != 0:
                score += 10 * newScaredTimes[i] * (newScaredTimes[i] - ghostNewDist)
            elif ghostNewDist != 0:
                print("ghost dist ", ghostNewDist)
                score -= 50 / ghostNewDist
            elif ghostNewDist == 0: # dont have it move on top of ghost
                score -= 5000

        new_closest_food = [-1, -1]
        new_closest_dist = 9999999

        for i in range(newFood.width):
            for j in range(newFood.height):


                if curFood[i][j] and manhattanDistance(newPos, (i, j)) < new_closest_dist:
                    new_closest_dist = manhattanDistance(newPos, (i, j))
                    new_closest_food = [i, j]

        if new_closest_dist == 0:
            score += 30
        else:
            score += 20 / new_closest_dist

        print("new score: ", score, " close dist ", new_closest_dist, " food", new_closest_food, "pacman", newPos)

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        value, action = self.scoreGameState(gameState, 0, self.depth * gameState.getNumAgents())
        return action

    def scoreGameState(self, gameState, player, depth):
        """
        Returns the minimax score given a gamestate, a start player, and an initial depth
        """

        # Evaluates if it reached an end state or went through max depth
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), None

        ret_val = 0
        best_action = None
        if player == 0:  # Searches for the max state
            max = -9999999

            for action in gameState.getLegalActions():
                value, temp = self.scoreGameState(gameState.generateSuccessor(player, action), (player + 1) % gameState.getNumAgents(), depth - 1)
                if value > max:
                    max = value
                    best_action = action
            ret_val = max
        else:  # Searches for the min state
            min = 9999999
            for action in gameState.getLegalActions(player):
                value, temp = self.scoreGameState(gameState.generateSuccessor(player, action), (player + 1) % gameState.getNumAgents(), depth - 1)
                if value < min:
                    min = value
            ret_val = min

        return ret_val, best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        value, action = self.scoreGameState(gameState, 0, self.depth * gameState.getNumAgents())
        return action


    def scoreGameState(self, gameState, player, depth, alpha= -9999999, beta= 9999999):
        """
        Returns the minimax score given a gamestate, a start player, and an initial depth
        """

        # Evaluates if it reached an end state or went through max depth
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), None

        best_action = None
        if player == 0:  # Searches for the max state
            for action in gameState.getLegalActions():
                value, temp = self.scoreGameState(gameState.generateSuccessor(player, action), (player + 1) % gameState.getNumAgents(), depth - 1, alpha, beta)
                if value > alpha:
                    alpha = value
                    best_action = action
                if alpha >= beta:
                    return beta, best_action
            return alpha, best_action
        else:  # Searches for the min state
            for action in gameState.getLegalActions(player):
                value, temp = self.scoreGameState(gameState.generateSuccessor(player, action), (player + 1) % gameState.getNumAgents(), depth - 1, alpha, beta)
                if value < beta:
                    beta = value
                if beta <= alpha:
                    return alpha, None
            return beta, None


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        value, action = self.scoreGameState(gameState, 0, self.depth * gameState.getNumAgents())
        return action

    def scoreGameState(self, gameState, player, depth):
        """
        Returns the minimax score given a gamestate, a start player, and an initial depth
        """

        # Evaluates if it reached an end state or went through max depth
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState), None

        ret_val = 0
        best_action = None
        if player == 0:  # Searches for the max state
            max = -9999999

            for action in gameState.getLegalActions():
                value, temp = self.scoreGameState(gameState.generateSuccessor(player, action), (player + 1) % gameState.getNumAgents(), depth - 1)
                if value > max:
                    max = value
                    best_action = action
            ret_val = max
        else:  # Searches for the min state
            score = 0
            actions = 0
            for action in gameState.getLegalActions(player):
                value, temp = self.scoreGameState(gameState.generateSuccessor(player, action), (player + 1) % gameState.getNumAgents(), depth - 1)
                score += value
                actions += 1
            ret_val = score / actions

        return ret_val, best_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
