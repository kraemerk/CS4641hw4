import sys
sys.path.append('./burlap.jar')

import java

from burlap.behavior.singleagent.auxiliary import StateReachability
from burlap.behavior.singleagent.learning.actorcritic import ActorCritic
from burlap.domain.singleagent.gridworld import GridWorldDomain
from burlap.domain.singleagent.gridworld import GridWorldVisualizer
from burlap.domain.singleagent.gridworld.state import GridAgent
from burlap.domain.singleagent.gridworld.state import GridLocation
from burlap.domain.singleagent.gridworld.state import GridWorldState
from burlap.behavior.policy import PolicyUtils, BoltzmannQPolicy
from burlap.statehashing.simple import SimpleHashableStateFactory
from burlap.mdp.singleagent import SADomain
from burlap.mdp.singleagent.environment import SimulatedEnvironment
from burlap.behavior.singleagent.auxiliary.performance import LearningAlgorithmExperimenter
from burlap.mdp.core.state import State
from burlap.shell.visual import VisualExplorer
from burlap.visualizer import Visualizer
from burlap.mdp.core.oo.state import ObjectInstance
from burlap.domain.singleagent.gridworld import GridWorldRewardFunction
from burlap.mdp.singleagent.model import RewardFunction
from burlap.domain.singleagent.gridworld import GridWorldTerminalFunction
from burlap.behavior.singleagent.auxiliary.valuefunctionvis import ValueFunctionVisualizerGUI
from burlap.behavior.singleagent.planning.stochastic.valueiteration import ValueIteration
from burlap.mdp.auxiliary.stateconditiontest import TFGoalCondition
from burlap.mdp.singleagent.common import VisualActionObserver
from burlap.behavior.singleagent.auxiliary import EpisodeSequenceVisualizer
from burlap.behavior.singleagent.planning.stochastic.policyiteration import PolicyIteration;
from burlap.behavior.singleagent.learning.tdmethods import QLearning
from burlap.behavior.policy import GreedyQPolicy
from burlap.behavior.singleagent.auxiliary.performance import PerformancePlotter, TrialMode, PerformanceMetric
from burlap.behavior.singleagent.learning import LearningAgentFactory
from collections import defaultdict
from time import clock
import csv

env = None

##
# Control switches
##
MODE = 'easy' # easy / hard
VISUALIZE_GRID_WORLD = True
PERFORM_VALUE_ITERATION = True
PERFORM_POLICY_ITERATION = True
PERFORM_Q_LEARNING_ITERATION = True

class SimpleRewardFunction(RewardFunction):
	def __init__(self, goalX, goalY, gridMap):
		self.goalX = goalX
		self.goalY = goalY
		self.map = gridMap

	def reward(self, state, action, sprime):
		a = sprime.agent
		ax = a.x
		ay = a.y

		if ax == self.goalX and ay == self.goalY:
			return 100.;

		if self.map[ax][ay] < 0:
			return self.map[ax][ay]*10

		return -1;

def getEasyGrid():
	return ([[0, 1, 0, 0, 0, 0, 1],# 12 full
			 [0, 1, 0, 0, 1, 0, 0],
			 [0, 1, 0, 0, 1, 0, 0],
			 [0, 0, 0, 1, 1, 0, 0],
			 [0, 0, 0, 1, 0, 0, 0],
			 [0, 1, 0, 0, 0, 0, 1],
			 [0, 0, 0, 0, 1, 0, 0]], [5, 6])

def getHardGrid(): #122 full
	return ([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
			 [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
			 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
			 [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1],
			 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
			 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
			 [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
			 [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
			 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
			 [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
			 [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0],
			 [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
			 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
			 [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
			 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
			 [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
			 [0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
			 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
			 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0],
			 [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0],
			 [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
			 [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
			 [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
			 [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
			 [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], [24, 24])

def writeCSV(iters, fileName, times, rewards, steps):
	with open(fileName, 'wb') as file:
		file.write('iter,time,reward,steps\n')
		writer = csv.writer(file, delimiter=',')
		writer.writerows(zip(iters, times, rewards, steps))

def runEvals(i, e, startTime):
	reward = sum(e.rewardSequence)
	steps = e.numTimeSteps()
	print("{} - {} reward @ {} steps".format(i, reward, steps))
	return (reward, steps, clock() - startTime)

def simpleValueFunctionVis(valueFunc, maxX, maxY, policy, state, domain, hashingFactory, title):
	states = StateReachability.getReachableStates(state, domain, hashingFactory)
	gui = GridWorldDomain.getGridWorldValueFunctionVisualization(states, maxX, maxY, valueFunc, policy);
	gui.setTitle(title);
	gui.initGUI();

def visualizeGridWorld(domain, gw, state):
	v = GridWorldVisualizer.getVisualizer(gw.getMap())

	exp = VisualExplorer(domain, v, state)

	exp.addKeyAction(",", GridWorldDomain.ACTION_NORTH, "")
	exp.addKeyAction("a", GridWorldDomain.ACTION_WEST, "")
	exp.addKeyAction("o", GridWorldDomain.ACTION_SOUTH, "")
	exp.addKeyAction("e", GridWorldDomain.ACTION_EAST, "")

	# exp.addKeyAction("w", GridWorldDomain.ACTION_NORTH, "")
	# exp.addKeyAction("a", GridWorldDomain.ACTION_WEST, "")
	# exp.addKeyAction("s", GridWorldDomain.ACTION_SOUTH, "")
	# exp.addKeyAction("d", GridWorldDomain.ACTION_EAST, "")

	exp.initGUI()

def writePolicyPath(name, p, state, domain):
	outputPath = "./output/{}".format(name)
	PolicyUtils.rollout(p, state, domain.getModel()).write(outputPath);
	return outputPath

def visualizeOutputPath(path, gw, domain):
	v = GridWorldVisualizer.getVisualizer(gw.getMap())
	EpisodeSequenceVisualizer(v, domain, './output/')

if __name__ == '__main__':

	print("-=-=-=-=- Initializing %s Grid World -=-=-=-=-\n" % MODE)

	discount = 0.99
	MAX_ITERATIONS = 100
	MAX_Q_ITERATIONS = 1000

	gridMap = None
	goal = None
	if MODE == 'easy':
		gridMap = getEasyGrid()[0]
		goal = getEasyGrid()[1]
	elif MODE == 'hard':
		gridMap = getHardGrid()[0]
		goal = getHardGrid()[1]
	else:
		print("ERROR UNKNOWN MODE: %s" % MODE)
	n = len(gridMap[0])


	gw = GridWorldDomain(n,n) # n x n grid world

	for row, cells in enumerate(gridMap):
		for column, value in enumerate(cells):
			if value == 1:
				gw.setCellWallState(column, n - 1 - row, 1)

	tf = GridWorldTerminalFunction(goal[0], goal[1])
	gw.setTf(tf)

	rf = SimpleRewardFunction(goal[0], goal[1], gridMap)
	gw.setRf(rf)

	goalCondition = TFGoalCondition(tf)

	# gw.setMapToFourRooms() #four rooms layout
	gw.setProbSucceedTransitionDynamics(0.80) #stochastic transitions with 0.80 success rate
	domain = gw.generateDomain() # generate the grid world domain

	state = GridWorldState(GridAgent(0, 0), [GridLocation(goal[0], goal[1], "loc0")])

	env = SimulatedEnvironment(domain, state)

	# observer = VisualActionObserver(domain, GridWorldVisualizer.getVisualizer(gw.getMap()))
	# env.addObservers(observer)
	# observer.initGUI()

	if VISUALIZE_GRID_WORLD:
		visualizeGridWorld(domain, gw, state)

	if PERFORM_VALUE_ITERATION:
		print("-=-=-=-=- Value Iteration (%s) Analysis -=-=-=-=-\n" % MODE)

		hashingFactory = SimpleHashableStateFactory()

		rewards = []
		steps = []
		times = []

		for i in range(1, MAX_ITERATIONS + 1):
			startTime = clock()

			planner = ValueIteration(domain, discount, hashingFactory, 0.001, MAX_ITERATIONS)
				# planner.toggleDebugPrinting(False)
			plan = planner.planFromState(state)
			print("Time: %s" % (clock() - startTime))
			episode = PolicyUtils.rollout(plan, env)

			# results = runEvals(1, episode, startTime)
			# print("RewardSequence: %s" % results[0])

			results = runEvals(i, episode, startTime)
			rewards.append(results[0])
			steps.append(results[1])
			times.append(results[2])

			if i == 1:
				simpleValueFunctionVis(planner, n, n, plan, state, domain, hashingFactory, "Value Iteration {}".format(i))
				# outputPath = writePolicyPath('value_iteration_1', plan, state, domain)
				# visualizeOutputPath(outputPath, gw, domain)

			if i == 100:
				simpleValueFunctionVis(planner, n, n, plan, state, domain, hashingFactory, "Value Iteration {}".format(i))
				# outputPath = writePolicyPath('value_iteration_100', plan, state, domain)
				# visualizeOutputPath(outputPath, gw, domain)

		writeCSV(range(1, MAX_ITERATIONS + 1), 'ValueIteration (%s).csv' % MODE, times, rewards, steps)

	if PERFORM_POLICY_ITERATION:
		print("-=-=-=-=- Policy Iteration (%s) Analysis -=-=-=-=-\n" % MODE)

		hashingFactory = SimpleHashableStateFactory()

		rewards = []
		steps = []
		times = []

		start = 4

		for i in range(1, MAX_ITERATIONS + 1):
			startTime = clock()

			planner = PolicyIteration(domain, discount, hashingFactory, 0.001, 1, i)
			# planner.toggleDebugPrinting(False)
			plan = planner.planFromState(state);
			print("Time: %s" % (clock() - startTime))
			episode = PolicyUtils.rollout(plan, state, domain.getModel())

			results = runEvals(i, episode, startTime)
			rewards.append(results[0])
			steps.append(results[1])
			times.append(results[2])

			if i == 1:
				simpleValueFunctionVis(planner, n, n, plan, state, domain, hashingFactory, "Policy Iteration {}".format(i))
				# outputPath = writePolicyPath('policy_iteration_1', plan, state, domain)
				# visualizeOutputPath(outputPath, gw, domain)

			if i == 100:
				simpleValueFunctionVis(planner, n, n, plan, state, domain, hashingFactory, "Policy Iteration {}".format(i))
				# outputPath = writePolicyPath('value_iteration_100', plan, state, domain)
				# visualizeOutputPath(outputPath, gw, domain)

		writeCSV(range(1, MAX_ITERATIONS + 1), 'PolicyIteration (%s).csv' % MODE, times, rewards, steps)

	if PERFORM_Q_LEARNING_ITERATION:
		print("-=-=-=-=- Q LEARNING (%s) Analysis -=-=-=-=-\n" % MODE)

		hashingFactory = SimpleHashableStateFactory()

		rewards = []
		steps = []
		times = []

		startTime = clock()

		agent = QLearning(domain, discount, hashingFactory, 0., 0.1)

		for i in range(1, MAX_Q_ITERATIONS + 1):
			episode = agent.runLearningEpisode(env)
			results = runEvals(i, episode, startTime)
			rewards.append(results[0])
			steps.append(results[1])
			times.append(results[2])

			env.resetEnvironment()

		agent.initializeForPlanning(1)
		p = agent.planFromState(state)

		print("Time: %s" % (clock() - startTime))

		simpleValueFunctionVis(agent, n, n, p, state, domain, hashingFactory, "Q Learning {}".format(1000))
		writeCSV(range(1, MAX_Q_ITERATIONS + 1), 'QLearning (%s).csv' % MODE, times, rewards, steps)

		# startTime = clock()
		# agent = QLearning(domain, discount, hashingFactory, 0., 1.)
		# agent.initializeForPlanning(10000)
		# plan = agent.planFromState(state)
		# print('Time: %s' % (clock() - startTime))
		# episode = agent.runLearningEpisode(env, 4)

		# qlearningfactory = QLearningFactory()

		# for i in range(1, MAX_Q_ITERATIONS + 1):
		# 	startTime = clock()

		# 	episode = agent.runLearningEpisode(env)
		# 	episode.write('./output/QL_{}'.format(i))
		# 	# print('{}: {}'.format(i, episode.maxTimeStep()))
		# 	results = runEvals(i, episode, startTime)
		# 	rewards.append(results[0])
		# 	steps.append(results[1])
		# 	times.append(results[2])

		# 	env.resetEnvironment()

		# 	if i == 100:
		# simpleValueFunctionVis(agent, n, n, GreedyQPolicy(agent), state, domain, hashingFactory, "Q Learning {}".format(1000))


		# writeCSV(range(1, MAX_Q_ITERATIONS + 1), 'QLearning (%s).csv' % MODE, times, rewards, steps)

		class QLearningFactory(LearningAgentFactory):
			# def __init__(self):

			def getAgentName(self):
				return 'Q-Learning'

			def generateAgent(self):
				return QLearning(domain, discount, hashingFactory, 0., 0.1)

		# exp = LearningAlgorithmExperimenter(env, 10, 1500, [QLearningFactory()]);
		# exp.setUpPlottingConfiguration(500, 250, 2, 1000, TrialMode.MOST_RECENT_AND_AVERAGE, PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE, PerformanceMetric.AVERAGE_EPISODE_REWARD, PerformanceMetric.MEDIAN_EPISODE_REWARD);

		# exp.startExperiment();

	print("-=-=-=-=- Finished -=-=-=-=-\n")



