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


from pkg_resources import evaluate_marker
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        legalMoves.remove("Stop")  # my code, stopping is always bad

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()

        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        distancesToFood = [
            util.manhattanDistance(newPos, food) for food in newFood.asList()
        ]
        distanceToNearestFood = min(distancesToFood) if len(distancesToFood) != 0 else 0
        distanceToFurthestFood = (
            max(distancesToFood) if len(distancesToFood) != 0 else 0
        )

        evaluation = score - ((distanceToNearestFood / (distanceToFurthestFood + 1)))

        return evaluation


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax(self, gameState: GameState, depth, agentIndex=0):
        legalActions = gameState.getLegalActions(agentIndex)

        if (
            depth == 0
            or len(legalActions) == 0
            or gameState.isWin()
            or gameState.isLose()
        ):
            return None, self.evaluationFunction(gameState)

        if agentIndex == 0:
            evaluations = [-float("inf") for _ in legalActions]

            for i in range(len(legalActions)):
                successor = gameState.generateSuccessor(agentIndex, legalActions[i])
                _, evaluations[i] = self.minimax(successor, depth, agentIndex + 1)

            evaluation_index = evaluations.index(max(evaluations))
        else:
            evaluations = [float("inf") for _ in legalActions]

            nextAgentIndex = agentIndex + 1
            if nextAgentIndex == gameState.getNumAgents():
                nextAgentIndex = 0
            if nextAgentIndex == 0:
                depth -= 1

            for i in range(len(legalActions)):
                successor = gameState.generateSuccessor(agentIndex, legalActions[i])
                _, evaluations[i] = self.minimax(successor, depth, nextAgentIndex)

            evaluation_index = evaluations.index(min(evaluations))

        action = legalActions[evaluation_index]
        evaluation = evaluations[evaluation_index]

        return action, evaluation

    def getAction(self, gameState: GameState):
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
        "*** YOUR CODE HERE ***"

        # maximizer is pacman, minimizer is ghost
        action, _ = self.minimax(gameState, depth=self.depth, agentIndex=0)

        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def minimax(self, gameState: GameState, depth, agentIndex, alpha, beta):
        legalActions = gameState.getLegalActions(agentIndex)

        if (
            depth == 0
            or len(legalActions) == 0
            or gameState.isWin()
            or gameState.isLose()
        ):
            return None, self.evaluationFunction(gameState)

        if agentIndex == 0:
            evaluation = -float("inf")
            action = None

            for legalAction in legalActions:
                successor = gameState.generateSuccessor(agentIndex, legalAction)
                _, child_evaluation = self.minimax(
                    successor, depth, agentIndex + 1, alpha, beta
                )

                if child_evaluation > evaluation:
                    evaluation = child_evaluation
                    action = legalAction

                if evaluation > beta:
                    break
                alpha = max(alpha, evaluation)

        else:
            evaluation = float("inf")
            action = None

            nextAgentIndex = agentIndex + 1
            if nextAgentIndex == gameState.getNumAgents():
                nextAgentIndex = 0
            if nextAgentIndex == 0:
                depth -= 1

            for legalAction in legalActions:
                successor = gameState.generateSuccessor(agentIndex, legalAction)
                _, child_evaluation = self.minimax(
                    successor, depth, nextAgentIndex, alpha, beta
                )

                if child_evaluation < evaluation:
                    evaluation = child_evaluation
                    action = legalAction

                if evaluation < alpha:
                    break
                beta = min(beta, evaluation)

        return action, evaluation

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        action, _ = self.minimax(
            gameState,
            depth=self.depth,
            agentIndex=0,
            alpha=-float("inf"),
            beta=float("inf"),
        )

        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def expectimax(self, gameState: GameState, depth, agentIndex):
        legalActions = gameState.getLegalActions(agentIndex)

        if (
            depth == 0
            or len(legalActions) == 0
            or gameState.isWin()
            or gameState.isLose()
        ):
            return None, self.evaluationFunction(gameState)

        if agentIndex == 0:
            evaluation = -float("inf")
            action = None

            for legalAction in legalActions:
                successor = gameState.generateSuccessor(agentIndex, legalAction)
                _, child_evaluation = self.expectimax(successor, depth, agentIndex + 1)

                if child_evaluation > evaluation:
                    evaluation = child_evaluation
                    action = legalAction

        else:
            evaluation = 0
            action = None

            nextAgentIndex = agentIndex + 1
            if nextAgentIndex == gameState.getNumAgents():
                nextAgentIndex = 0
            if nextAgentIndex == 0:
                depth -= 1

            for legalAction in legalActions:
                successor = gameState.generateSuccessor(agentIndex, legalAction)
                _, child_evaluation = self.expectimax(successor, depth, nextAgentIndex)
                evaluation += child_evaluation

            evaluation /= len(legalActions)

        return action, evaluation

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, _ = self.expectimax(
            gameState,
            depth=self.depth,
            agentIndex=0,
        )

        return action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <evaluation is power>
    """
    "*** YOUR CODE HERE ***"

    score = currentGameState.getScore()
    position = currentGameState.getPacmanPosition()
    ghostPositions = currentGameState.getGhostPositions()
    foodPositions = currentGameState.getFood().asList()

    distancesToFood = [util.manhattanDistance(position, food) for food in foodPositions]
    distancesToGhosts = [
        util.manhattanDistance(position, ghostPosition)
        for ghostPosition in ghostPositions
    ]
    distanceToNearestFood = min(distancesToFood) if len(distancesToFood) != 0 else 0
    distanceToFurthestFood = max(distancesToFood) if len(distancesToFood) != 0 else 0

    distanceToNearestGhost = (
        min(distancesToGhosts) if len(distancesToGhosts) != 0 else 0
    )
    distanceToFurthestGhost = (
        max(distancesToGhosts) if len(distancesToGhosts) != 0 else 0
    )

    evaluation = (
        score
        - ((distanceToNearestFood / (distanceToFurthestFood + 1)) * 1.5)
        + ((distanceToNearestGhost / (distanceToFurthestGhost + 1)) * 1.2)
    )

    return evaluation


# Abbreviation
better = betterEvaluationFunction
