import gym
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import math
import numpy as np
import random

#GAME = 'CartPole-v0'
GAME = 'MountainCar-v0'

class Environment:
	def __init__(self, game):
		self.game = game
		self.env = gym.make(game)
		
	def run(self, agent):
		s = self.env.reset()
		total_r = 0
		
		while True:
			self.env.render()
			a = agent.act(s)
			s_, r, done, info = self.env.step(a)
			if done:
				s_ = None
			agent.observe((s, a, r, s_))
			agent.replay()
			
			s = s_
			total_r += r
			
			if done:
				break
				
		print("Total reward:",total_r)
		
		


MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99

MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.001

		
class Agent:
	steps = 0
	epsilon = MAX_EPSILON
	
	def __init__(self, stateCnt, actionCnt):
		self.stateCnt = stateCnt
		self.actionCnt = actionCnt
		self.brain = Brain(stateCnt, actionCnt)
		self.memory = Memory(MEMORY_CAPACITY)
		
	def act(self, s):
		if random.random() < self.epsilon:
			return random.randint(0, self.actionCnt-1)
		else:
			return np.argmax(self.brain.predictOne(s))
			
	def observe(self, sample):
		self.memory.add(sample)
		# slowly decrease Epsilon based on our eperience
		self.steps += 1
		self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
		
	def replay(self):
		batch = self.memory.sample(BATCH_SIZE)
		
		bat_len = len(batch)
		
		none_state = np.zeros(self.stateCnt)
		
		states = np.array([o[0] for o in batch])
		states_ = np.array([ (none_state if o[3] is None else o[3]) for o in batch ])

		p = self.brain.predict(states)
		p_ = self.brain.predict(states_)
		
		x = np.zeros((bat_len, self.stateCnt))
		y = np.zeros((bat_len, self.actionCnt))
		
		for i in range(bat_len):
			o = batch[i]
			#print ("batch value", o)
			s = o[0]
			a = o[1]
			r = o[2]
			s_ = o[3]
			
			t = p[i]
			#print ("t value", t)
			
			if s_ is None:
				t[a] = r
			else:
				t[a] = r + GAMMA * np.amax(p_[i])
				
			x[i] = s
			y[i] = t
			
		self.brain.train(x, y)
		
class Brain:
	def __init__(self, stateCnt, actionCnt):
		self.stateCnt = stateCnt
		self.actionCnt = actionCnt
		
		self.model = self.buildModel()
		
	def buildModel(self):
		
		input_size = self.stateCnt
		
		layer = input_data(shape=[None, 1, input_size], name = 'input')

		layer = fully_connected(layer, 64, activation = 'relu')
		#layer = dropout(layer, 0.8)

		layer = fully_connected(layer, self.actionCnt, activation = 'linear')

		layer = regression(layer, optimizer= 'RMSProp', learning_rate = 0.00025, loss = 'mean_square',name = 'targets')
		model = tflearn.DNN(layer, tensorboard_dir = 'log')
		
		return model
		
	def train(self, x, y, epoch = 1):
		x = x.reshape(-1, 1, self.stateCnt)
		#print ("input to train", x.shape)
		return self.model.fit(x, y, batch_size = 64, n_epoch = epoch, show_metric = False, run_id = 'game_learning')
		
	def predict(self, s):
		s = s.reshape(-1, 1, self.stateCnt)
		#print ("input to predict",s.shape)
		return self.model.predict(s)
		
	def predictOne(self, s):
		return self.predict(s.reshape(1, self.stateCnt))#.flatten()
		
	
		
		
class Memory:
	samples = []

	def __init__(self, capacity):
		self.capacity = capacity

	def add(self, sample):
		self.samples.append(sample)		   

		if len(self.samples) > self.capacity:
			self.samples.pop(0)

	def sample(self, n):
		n = min(n, len(self.samples))
		return random.sample(self.samples, n)
		
if __name__ == "__main__":
	
	E = Environment(GAME)
	stateCnt = E.env.observation_space.shape[0]
	actionCnt = E.env.action_space.n
	print(stateCnt)
	print(actionCnt)
	agent = Agent(stateCnt, actionCnt)
	
	try:
		while True:
			E.run(agent)
			#break
			
	finally:
		agent.brain.model.save(GAME + "_model.h5")
