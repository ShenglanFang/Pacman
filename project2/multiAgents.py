# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

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
    some Directions.X for some X in the set {North, South, West, East, Stop}
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
    successorGameState = currentGameState.generatePacmanSuccessor(action) #the game states returned
    newPos = successorGameState.getPacmanPosition() # tuple returned indicating the successor position
    newFood = successorGameState.getFood()    # check by if newFood[x][y] == True:
    newGhostStates = successorGameState.getGhostStates()
    foodPositions = newFood.asList()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    ghostScore= 0
    ghostPosition = newGhostStates[0].getPosition()
    ghostScore += abs(newPos[0]-ghostPosition[0]) + abs(newPos[1]-ghostPosition[1])
    currentScore = 0
    foodScore = 9999
    foodPositionsCopy = foodPositions
    for foodPosition in foodPositionsCopy:
      currentScore = abs(newPos[0]-foodPosition[0]) + abs(newPos[1]-foodPosition[1])
      if currentScore < foodScore:
        foodScore = currentScore
    capsuleScore = 0
    capsulePoss = currentGameState.getCapsules()
    for capsule in capsulePoss:
      if newPos == capsule:
        capsuleScore = 1000
    stopScore = 0
    if action == "Stop":
      stopScore = -50
    return successorGameState.getScore() + ghostScore/ foodScore + capsuleScore + stopScore

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

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    actionState = self.value(gameState, 0, 0)
    action = actionState[1]
    return action

  def value(self, currentGameState, agentIndex, depth):
    if depth % 2 == 0:
      return self.maxValue(currentGameState, depth)
    else:
      return self.minValue(currentGameState, agentIndex, depth)

  def maxValue(self, currentGameState, depth):
    if depth == self.depth or currentGameState.isWin() or currentGameState.isLose():
      return scoreEvaluationFunction(currentGameState)
    score = -999999999
    scoreCopy=score
    validMove =""
    legalMoves = currentGameState.getLegalActions()
    if "Stop" in legalMoves:
      legalMoves.remove("Stop")
    for move in legalMoves:
      successor = currentGameState.generatePacmanSuccessor(move)
      score = max(score, self.minValue(successor, 1, depth))
      if scoreCopy != score:
        validMove = move
        scoreCopy = score
    return score, validMove

  def minValue(self, currentGameState, agentIndex, depth):
    if depth == self.depth or currentGameState.isWin() or currentGameState.isLose():
      return scoreEvaluationFunction(currentGameState)
    score = 999999999
    scoreCopy = score
    validMove =""
    legalMoves = currentGameState.getLegalActions(agentIndex)
    if "Stop" in legalMoves:
      legalMoves.remove("Stop")
    for move in legalMoves:
      successor = currentGameState.generateSuccessor(agentIndex, move)
      if agentIndex < currentGameState.getNumAgents()-1:
        score = min(score, self.minValue(successor, agentIndex+1, depth))
      else:
        score = min(score, self.maxValue(successor, depth+1))
      if scoreCopy != score:
        validMove = move
        scoreCopy = score
    return score, validMove

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    return self.maxValue(gameState, 0, 0, -9999999, 9999999)[1]

  def value(self, gameState, nextAgentIndex, depth, alpha, beta):
      if gameState.isWin() or gameState.isLose() or depth >= gameState.getNumAgents() * self.depth:
        return self.evaluationFunction(gameState)
      if nextAgentIndex == 0:
          return (self.maxValue(gameState, nextAgentIndex, depth, alpha, beta))[0]
      else:
          return (self.minValue(gameState, nextAgentIndex, depth, alpha, beta))[0]
      
  def maxValue(self, gameState, agentIndex, depth, alpha, beta):
      v = -9999999
      bestAction = 'Stop'
      # for each successor of state:
      # v = max(v, value(successor, alpha, beta))
      # if v >= beta return v
      # alpha = max(alpha, v)
      legalActions = gameState.getLegalActions(agentIndex)
      for action in legalActions:
          successorState = gameState.generateSuccessor(agentIndex, action)
          nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents();
          valueSuccessorState = self.value(successorState, nextAgentIndex, depth + 1, alpha, beta)
          if v < valueSuccessorState:
              v = valueSuccessorState
              bestAction = action
          if v > beta:
              return (v, bestAction)
          alpha = max(alpha, v)
      return (v, bestAction)

  def minValue(self, gameState, agentIndex, depth, alpha, beta):
      v = 9999999
      bestAction = 'Stop'
      # for each successor of state:
      # v = min(v, value(successor, alpha, beta))
      # if v <= alpha return v
      # beta = min(beta, v)
      legalActions = gameState.getLegalActions(agentIndex)
      for action in legalActions:
          successorState = gameState.generateSuccessor(agentIndex, action)
          nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents();
          valueSuccessorState = self.value(successorState, nextAgentIndex, depth + 1, alpha, beta)
          if v > valueSuccessorState:
              v = valueSuccessorState
              bestAction = action
          if v < alpha:
              return (v, bestAction)
          beta = min(beta, v)
      return (v, bestAction)


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
    "*** YOUR CODE HERE ***"
    actions = gameState.getLegalActions()
    if "Stop" in actions:
      actions.remove("Stop")
    actionValue = 0
    resultMove = actions[0]
    for action in actions:
      currentValue = self.value(gameState.generatePacmanSuccessor(action), 1, 0)
      if currentValue > actionValue:
        actionValue = currentValue
        resultMove = action
    return resultMove

  def value(self, currentGameState, agentIndex, depth):
    if agentIndex == 0:
      return self.maxValue(currentGameState, depth)
    else:
      return self.expectValue(currentGameState, agentIndex, depth)

  def maxValue(self, currentGameState, depth):
    if depth == self.depth or currentGameState.isWin() or currentGameState.isLose():
      return self.evaluationFunction(currentGameState)
    score = -999999999
    legalMoves = currentGameState.getLegalActions()
    if "Stop" in legalMoves:
      legalMoves.remove("Stop")
    for move in legalMoves:
      successor = currentGameState.generatePacmanSuccessor(move)
      score = max(score, self.expectValue(successor, 1, depth))
    return score

  def expectValue(self, currentGameState, agentIndex, depth):
    if depth == self.depth or currentGameState.isWin() or currentGameState.isLose():
      return self.evaluationFunction(currentGameState)
    legalMoves = currentGameState.getLegalActions(agentIndex)
    if "Stop" in legalMoves:
      legalMoves.remove("Stop")
    probability = 1.0/len(legalMoves)
    score = 0
    for move in legalMoves:
      successor = currentGameState.generateSuccessor(agentIndex, move)
      if agentIndex < currentGameState.getNumAgents()-1:
        score += probability * self.expectValue(successor, agentIndex+1, depth)
      else:
        score += probability * self.maxValue(successor, depth+1)
    return score

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION:
      weighted linear sum of features:
      eval = sum(wi * fi)
      fearture1 : original getScore()
      fearture2 : the sum of pacman-ghost distance 
      fearture3 : the num of food remains
      fearture4 : the sum of ghost scaredTime
  """
  "*** YOUR CODE HERE ***"

  def feature1():
        return currentGameState.getScore()

  def feature2():
      pacmanPos = currentGameState.getPacmanPosition()
      ghostPoses = currentGameState.getGhostPositions()
      return sum([manhattanDistance(pacmanPos, ghostPos) for ghostPos in ghostPoses])

  def feature3():
      foodPoses = currentGameState.getFood()
      return len(list(foodPoses))

  def feature4():
      ghostStates = currentGameState.getGhostStates()
      scaredTime = [ghostState.scaredTimer for ghostState in ghostStates]
      return sum(scaredTime)

  weights = [1.0, -0.1, 0.6, 0.3]
  features = [feature1(), feature2(), feature3(), feature4()]
  evals = sum([ weights[i] * features[i] for i in range(4)])
  return evals

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """
  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    return self.maxValue(gameState, 0, 0, -9999999, 9999999)[1]

  def value(self, gameState, nextAgentIndex, depth, alpha, beta):
      if gameState.isWin() or gameState.isLose() or depth >= gameState.getNumAgents() * self.depth:
        return self.evaluationFunction(gameState)
      if nextAgentIndex == 0:
          return (self.maxValue(gameState, nextAgentIndex, depth, alpha, beta))[0]
      else:
          return (self.minValue(gameState, nextAgentIndex, depth, alpha, beta))[0]
      
  def maxValue(self, gameState, agentIndex, depth, alpha, beta):
      v = -9999999
      bestAction = 'Stop'
      # for each successor of state:
      # v = max(v, value(successor, alpha, beta))
      # if v >= beta return v
      # alpha = max(alpha, v)
      legalActions = gameState.getLegalActions(agentIndex)
      for action in legalActions:
          successorState = gameState.generateSuccessor(agentIndex, action)
          nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents();
          valueSuccessorState = self.value(successorState, nextAgentIndex, depth + 1, alpha, beta)
          if v < valueSuccessorState:
              v = valueSuccessorState
              bestAction = action
          if v > beta:
              return (v, bestAction)
          alpha = max(alpha, v)
      return (v, bestAction)

  def minValue(self, gameState, agentIndex, depth, alpha, beta):
      v = 9999999
      bestAction = 'Stop'
      # for each successor of state:
      # v = min(v, value(successor, alpha, beta))
      # if v <= alpha return v
      # beta = min(beta, v)
      legalActions = gameState.getLegalActions(agentIndex)
      for action in legalActions:
          successorState = gameState.generateSuccessor(agentIndex, action)
          nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents();
          valueSuccessorState = self.value(successorState, nextAgentIndex, depth + 1, alpha, beta)
          if v > valueSuccessorState:
              v = valueSuccessorState
              bestAction = action
          if v < alpha:
              return (v, bestAction)
          beta = min(beta, v)
      return (v, bestAction)


