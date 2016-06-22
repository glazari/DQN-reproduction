import gym
import numpy as np
from skimage import color
from skimage.transform import resize
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

def preprocess(obs):
	#max_frames = np.maximum.reduce(obs) #quantos frame???
	xyz = color.rgb2xyz(obs)
	y = xyz[:,:,:,1]	
	small = resize(y,(4,84,84))
	state = small.reshape(1, 4, 84, 84)	
	return state

def play_game(env, num_steps, render=False):
	total_rew = 0
	ob = env.reset()
	for t in range(num_steps):		
		a = act(env, ob) #the observation is of the previous m previous frames
		(ob, reward, done, _info) = env.step(a)
		total_rew += reward
		if render and t%3==0: env.render()
		if done: break
	return total_rew, t+1

def act(env, obs):
	return env.action_space.sample()

if __name__ == '__main__':
	
	env = gym.make('Pong-v0')		
	
	Total_Frames = 50000000
	Max_ep = 1000000
	num_steps = 5000
	#Lets assume the following parameters are the same
	
	##Initialize replay memory D to capacity N (1000000)
	D = list()
	##Initialize action value function with random with random weights
	print("creating Q network")	
	Q = Sequential()
	Q.add(Convolution2D(32, 8, 8, border_mode='valid', input_shape=(4,84,84)))
	Q.add(MaxPooling2D(pool_size=(4, 4)))
	Q.add(Dropout(0.5))
	Q.add(Activation('relu'))
	Q.add(Convolution2D(64, 4, 4, border_mode='valid'))
	Q.add(MaxPooling2D(pool_size=(2, 2)))
	Q.add(Dropout(0.5))
	Q.add(Activation('relu'))
	Q.add(Convolution2D(64, 3, 3, border_mode='valid'))
	Q.add(Dropout(0.5))
	Q.add(Activation('relu'))
	Q.add(Flatten())
	Q.add(Dense(512))
	Q.add(Activation('relu'))
	Q.add(Dense(6))
	Q.add(Activation('tanh')) #aqui o certo é um htan
	print("ok")	
	
	sgd = SGD()
	print("compiling Q network")	
	Q.compile(loss = 'mean_squared_error', optimizer = sgd)
	##Initialize target action-value function ^Q with same wieghts
	print("copying Q to Q_target")	
	Q_target = Q

	e = 0 #e-greedy policy, drops from e=1 to e=0.1	
	k = 4 #The agent sees and selects an action every kth frame	
	m = 4 #Number of frames looked at each moment
	replay_size = 1000000	

	##Populates replay memory with some random sequences
			
	##Starts Playing and training
	for episodes in range(Max_ep):
		##Initialize sequence game and sequence s1 pre-process sequence
		obs = np.zeros([m]+list(env.observation_space.shape))
		ob = env.reset()
		ob0 = ob 	#this is for fixing the even odd frames problem
		obs[0] = ob0
		State0 = preprocess(obs)
		treward = 0
		for t in range(num_steps):
			if t%k==0:			
				if np.random.rand() < e:
					action = env.action_space.sample()
				else:
					in_obs = preprocess(obs)				
					qval = Q.predict(in_obs, batch_size=1, verbose=0)				
					action = qval.argmax()
			(ob1, reward, done, _info) = env.step(action)
			ob = np.maximum.reduce([ob0,ob1])	#even odd frame
			ob0 = ob1 							#problem
			obs[1:m] = obs[0:m-1]
			obs[0] = ob
			treward += reward
			##Set State' = ob and preprocess State'
			State1 = preprocess(obs)
			D.insert(0,[State0,action,reward,State1])
			if len(D)>replay_size:
				D.pop()
			print(len(D))
			##Sample random minibatch of transitions from D
			batch = [D[i] for i in sorted(np.random.randint(0,len(D),32))]
			S0 = np.empty(State0.shape)
			A = []
			R = []
			S1 = np.empty(State0.shape)
			for example in batch:
				s0, a, r, s1 = example
				S0 = np.concatenate((S0,s0))
				A.append(a)
				R.append(R)
				S1 = np.concatenate((S1,s1))
			y_Q = Q.predict_on_batch(S0)
			y_Q_target = np.max(Q_target.predict_on_batch(S1),1)
			
			
				##perform gradient descent step on
			##Every C steps set ^Q = Q
			if done:break;	
		print("Episode",episodes+1,"\tpoints =",treward,"\tframes",t+1)


