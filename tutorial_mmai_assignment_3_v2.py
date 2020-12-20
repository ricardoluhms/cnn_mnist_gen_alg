
# %%
# imports
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# for the random seed
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.initializers import RandomUniform
from time import time
from random import randint

#tf.debugging.set_log_device_placement(False)

try:
	tf.device('/device:GPU')
except:
	tf.device('/device:CPU')

# set the random seeds to get reproducible results
print("#"*25, " Code Start", "#"*25,"\n")
np_seed = np.random.seed(1)
tf_seed = tf.random.set_seed(2)

# Load data from https://www.openml.org/d/554
print("#"*25, " Load Data", "#"*25,"\n")
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

X, y = X[:1000], y[:1000] # modified from 1000 to 5000
# reduces the dataset which has 70000 images to a smaller set

### deep learning is supervised 
#X = X.reshape(X.shape[0], 28, 28, 1) # rows, height, width, color channel
X = X.reshape(X.shape[0], 28, 28, 1) # rows, height, width, color channel
# Normalize
X = X / 255. # 8bit 2**8 =256 

# number of unique classes
num_classes = len(np.unique(y))
y = y.astype(int)
print("#"*25, " Split Data", "#"*25,"\n")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

num_tot = y.shape[0] # number images in the dataset
num_train = y_train.shape[0] #number of images for the training phase
num_test = y_test.shape[0] #number of images for the test phase

print("#"*25, " One Hot Encoding", "#"*25,"\n")
y_oh = np.zeros((num_tot, num_classes)) #generate a blank array to be filled with one-hotenc
y_oh[range(num_tot), y] = 1 # replace  

y_oh_train = np.zeros((num_train, num_classes))
y_oh_train[range(num_train), y_train] = 1

y_oh_test = np.zeros((num_test, num_classes))
y_oh_test[range(num_test), y_test] = 1

print("#"*25, " Questions Part I", "#"*25,"\n")
#### Question 1 Code Answers
for num, y_value in enumerate(y):
	if num<10:
		print(y_value, y_oh[num])

ax1 = plt.subplot(131) ### ( row=1 column=3 imgnumber=1 )
ax1.imshow(X[0]) ### X must be in the reshaped form (28,28,1) or (28,28) depeding on the matplot lib version
ax1.set_title( label = "Y label = "+ str( y[0] ))

ax2 = plt.subplot(132) ### ( row=1 column=1 imgnumber=2 )
ax2.imshow(X[10]) ### X must be in the reshaped form (28,28,1) or (28,28) depeding on the matplot lib version
ax2.set_title( label = "Y label ="+ str(y[10]))

ax3 = plt.subplot(133) ### ( row=1 column=1 imgnumber=3 )
ax3.imshow(X[20]) ### X must be in the reshaped form (28,28,1) or (28,28) depeding on the matplot lib version
ax3.set_title( label = "Y label ="+ str(y[20]))
plt.show()
# %%
print("#"*25, " Load CNN Class ", "#"*25,"\n")
class MyCNN():
	def __init__(self,
				 X_train, y_oh_train, X_test, y_oh_test, y_test,
				 activ_func = 'relu', last_layer_func = 'softmax' , ### Functions
				 standard_kernel = (3, 3),  input_shape=(28, 28, 1),  ### CNN input
				 num_classes = 10 ,                                   ### CNN output
				 min_image_kernels = 16, dropout_rate = 0.1,          ### CNN parameters
				 lr = 0.02, decay = 1e-6, momentum = 0.9,             ### Optimizer parameters
				 batch_size = 1000, epochs=1000,                       ### Batch and Epoch
				 loss='categorical_crossentropy', optimizer = "SGD"):   ### loss type

		self.X_train = X_train
		self.y_oh_train = y_oh_train
		self.X_test = X_test
		self.y_oh_test = y_oh_test
		self.y_test = y_test

		self.activ_func =  activ_func
		self.last_layer_func =  last_layer_func
		self.standard_kernel = standard_kernel
		self.input_shape = input_shape
		self.num_classes = num_classes
		self.dropout_rate = dropout_rate
		self.mik = min_image_kernels
		self.lr = lr
		self.decay=decay
		self.momentum=momentum
		self.batch_size = batch_size
		self.epochs = epochs
		
		self.optimizer = optimizer
		#self.initializer = tf.keras.initializers.Zeros()
		### under dev.  = initializer =RandomUniform(minval=0.9, maxval=1., seed=1)

	def create_model(self):

		self.model = Sequential()
		self.model.add( Conv2D( self.mik , (3, 3), 
							   activation= self.activ_func, 
							   input_shape= self.input_shape
							   ))

		### Conv layer 1 -  getting overall details
		### the more add number of filters/kernel the better, usually your network is
		### the higher the kernel shape you pick large part information

		# Max pooling
		self.model.add( MaxPooling2D ( pool_size = (2, 2) ) )
		self.model.add(Dropout(self.dropout_rate))

		self.model.add(Conv2D(self.mik * 2 , (3, 3), 
								activation = self.activ_func
								)) ### Conv layer 2 -  getting more details
		# Max pooling
		self.model.add( MaxPooling2D ( pool_size = (2, 2) ) ) ### resuming information  does not has weights

		self.model.add(Flatten())

		self.model.add(Dense( self.mik * 8, 
								activation = self.activ_func
								)) ### first hidden layer of the fully connected

		self.model.add(Dropout(self.dropout_rate))

		self.model.add(Dense(self.num_classes, activation=self.activ_func
								))

		self.sgd = SGD(lr = self.lr, decay = self.decay, momentum = self.momentum, nesterov=True) #####
		self.rmsp = RMSprop(learning_rate = self.lr, rho=0.9, momentum=0.0, epsilon= self.decay, centered=False)
		self.adag =Adagrad(learning_rate=self.lr, initial_accumulator_value=0.1, epsilon=self.decay)

		if self.optimizer == "SGD":
			optim = self.sgd
		elif self.optimizer == "RMSProp":
			optim = self.rmsp
		elif self.optimizer == "AdaGrad":
			optim = self.adag

		# Compile the model
		self.model.compile(loss='categorical_crossentropy', optimizer=optim )

	def train(self):
		start_time = time()
		self.history = self.model.fit(self.X_train, 
									  self.y_oh_train, 
									  batch_size= self.batch_size, 
									  epochs=self.epochs, verbose = 0,
									  validation_data = (self.X_test,self.y_oh_test)) ### 
		end_time = time()
		self.train_time = end_time - start_time

	def plot_train(self):
		plt.plot(self.history.history['loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.show()

	def test(self):
		# Evaluate performance
		self.test_loss = self.model.evaluate(self.X_test, self.y_oh_test, batch_size= self.batch_size)

		self.predictions_perc = self.model.predict(self.X_test, batch_size = self.batch_size)
		self.predictions_norm = np.argmax(self.predictions_perc, axis=1) 
		# change encoding again

		self.accuracy = ( (self.predictions_norm == self.y_test).sum() / self.predictions_norm.shape[0])
		print('Accuracy:', self.accuracy, "\n")

print("#"*25, " Load Genetic Algorithm Class ", "#"*25,"\n")
class Genetic_Algorithm():
	
	def __init__(self, 
		lr_ls, decay_ls, batch_size_ls, epoch_ls, dropout_ls, optimizer_ls,
		X_train, y_oh_train, X_test, y_oh_test,y_test, 
		num_agents = 8): 
	   
		### Available Hyperparameters - lr_ls, decay_ls, batch_size_ls, epoch_ls, dropout_ls
		### Input train and test data - X_train, y_oh_train, X_test, y_oh_test,y_test
		### Genetic Algorithm hyperparameters - num_agents = 8                               
		self.lr_ls = lr_ls
		self.decay_ls = decay_ls
		self.batch_size_ls = batch_size_ls
		self.epoch_ls = epoch_ls
		self.dropout_ls = dropout_ls
		self.optimizer_ls = optimizer_ls

		self.tt_gen_pos_01 = len(lr_ls)
		self.tt_gen_pos_02 = len(decay_ls)
		self.tt_gen_pos_03 = len(batch_size_ls)
		self.tt_gen_pos_04 = len(epoch_ls)
		self.tt_gen_pos_05 = len(dropout_ls)
		self.tt_gen_pos_06 = len(optimizer_ls)

		self.num_agents = num_agents

		self.X_train = X_train
		self.y_oh_train = y_oh_train
		self.X_test = X_test
		self.y_oh_test = y_oh_test
		self.y_test = y_test

		self.tested_policies = {}

	@staticmethod
	def convert_policy_2_name(policy):
		policy_name = str(policy)[1:-1]
		policy_name = policy_name.replace(", ", "-")
		return policy_name

	@staticmethod
	def convert_name_2_policy(policy_name):
		str_ls = policy_name.split("-")
		array = np.array(str_ls).astype(int)
		policy = array.tolist()
		return policy

	### under dev
	def store_policy_result (self, result):
		policy = result[0]
		# result = [policy, modelCNN.history, modelCNN.train_time, modelCNN.accuracy, score]

		policy_name = self.convert_policy_2_name(policy)

		if policy_name not in self.tested_policies.keys():
			self.tested_policies[policy_name] = [result]
		else:
			policy_results = self.tested_policies[policy_name]
			policy_results.append(result)
			self.tested_policies[policy_name] = policy_results
												
	def gen_random_policy(self):
		
		gen_pos_01 = randint(0, self.tt_gen_pos_01-1) ###learning_rate
		gen_pos_02 = randint(0, self.tt_gen_pos_02-1) ###decay_rate
		gen_pos_03 = randint(0, self.tt_gen_pos_03-1) ###batch_size
		gen_pos_04 = randint(0, self.tt_gen_pos_04-1) ###epochs
		gen_pos_05 = randint(0, self.tt_gen_pos_05-1) ###dropout_rate
		gen_pos_06 = randint(0, self.tt_gen_pos_06-1) ###dropout_rate
		policy = [gen_pos_01, gen_pos_02, gen_pos_03, gen_pos_04, gen_pos_05,gen_pos_06 ]
		### A policy is composed by index to retrieve values from:
		#   self.lr_ls          where policy[0] = lr
		#   self.decay_ls       where policy[1] = decay
		#   self.batch_size_ls  where policy[2] = batch_size
		#   self.epoch_ls       where policy[3] = epochs
		#   self.dropout_ls     where policy[4] = dropout_rate
		return policy
	
	def generate_single_agent (self, policy):

		lr_index = policy[0]
		decay_index = policy[1]
		bs_index = policy[2]
		ep_index = policy[3]
		dr_index = policy[4]
		opt_index = policy[5]
		
		modelCNN = MyCNN(self.X_train, self.y_oh_train, 
							self.X_test,  self.y_oh_test, self.y_test,
							dropout_rate = self.dropout_ls[dr_index],          
							lr = self.lr_ls[lr_index], 
							decay = self.decay_ls[decay_index],           
							batch_size = self.batch_size_ls[bs_index], 
							epochs=self.epoch_ls[ep_index],
							optimizer=self.optimizer_ls[opt_index])

		print('#'*3,' Model Training: \n')
		print('- optimizer : lr= {} decay= {}'.format( self.lr_ls[lr_index], self.decay_ls[decay_index]))
		print('- neurons : dropout_rate {}'.format(self.dropout_ls[dr_index]))
		print('- batch_size= {} epochs= {} '.format(self.batch_size_ls[bs_index], self.epoch_ls[ep_index]),'#'*3,"\n")

		modelCNN.create_model()
		modelCNN.train()
		modelCNN.test()
		#print(modelCNN.model.summary())
		result = [policy, modelCNN.history, modelCNN.train_time, modelCNN.accuracy]                
		return result

	def initialize_agents(self):
		self.overall_results = []
		
		for _ in range(self.num_agents):
			policy = self.gen_random_policy()
			result = self.generate_single_agent(policy)
			#[:,0]
			#if policy not in self.overall_results: ###
			self.overall_results.append(result)
			self.store_policy_result(result)
		
	def evaluate_policies(self):
		results_array = np.array(self.overall_results)
		accuracy_arr = results_array[:,3]
		self.overall_accuracy = accuracy_arr.mean()
		self.max_accuracy = accuracy_arr.max()

		time_arr = results_array[:,2]
		self.overall_time = time_arr.mean()
		self.min_time = time_arr.min()
		
		policy_score_sorter = {}
		sorted_scores = []
		for num, result in enumerate(self.overall_results):
			score = 0            
			
			if result[3] > 0.5:
				score += 10

			if result[3] >= self.overall_accuracy:
				if result[3] == self.max_accuracy:
					score += 20

				else:
					score += 3
			
			if result[2] <= self.overall_time:
				if result[2] == self.min_time:
					score += 10
				else:
					score += 5

			policy_score_sorter[num] = score 

		sorted_scores = sorted(policy_score_sorter.items(), 
								 key=lambda kv: kv[1], 
								 reverse=True)
		sorted_policies = []                    
		### sorted_policies [(score, policy)] it will sort from highest to lowest
		for policy_num, score in sorted_scores:
			sorted_policies.append(self.overall_results[policy_num])
		
		self.overall_results = sorted_policies
	
	def agent_selection (self):
		
		self.parents = []
		### Make sure odd numbers are not used
		### The last agent with the worst performance is droppped
		if len(self.overall_results)%2 != 0:
			self.overall_results.pop(-1)
		### max pairs is the number of existing agents divided by four
		### since each pair generates 2 offsprings 
		### we will keep the same number of agents to avoid exponential growth of agent population
		max_pairs = int(len(self.overall_results)//4)
		possible_parents = len(self.overall_results)-1
		count = 0

		while len(self.parents) < max_pairs:

			if count < max_pairs:

				parent1 = count
				parent2 = np.random.choice(possible_parents, size=((1)))[0]
				parents_codes = np.array([parent1,parent2])

			else:
				parents_codes = np.random.choice(possible_parents, size=((2)))

			if parents_codes[0] != parents_codes[1]:

				parent1 = parents_codes[0]
				parent2 = parents_codes[1]

				policy1 = np.array(self.overall_results[parent1][0])
				policy2 = np.array(self.overall_results[parent2][0])
				#
				self.parents.append([policy1,policy2])
				count += 1

	def crossover(self, policy1, policy2,genes_pos = [0, 2, 4]):
		'''
		Arguments
		----------
		policy1: parent 1
		policy2: parent 2
		self.tt_gen_pos_01 = len(lr_ls)
		self.tt_gen_pos_02 = len(decay_ls)
		self.tt_gen_pos_03 = len(batch_size_ls)
		self.tt_gen_pos_04 = len(epoch_ls)
		self.tt_gen_pos_05 = len(dropout_ls)
		Return
		--------
		new_policy: offspring
		'''
		policyX = policy1.copy()
		policyY = policy2.copy()
		for g_pos in genes_pos:
			#from IPython import embed; embed()
			slice_policyX = policyX[g_pos]
			slice_policyY = policyY[g_pos]

			policyY[g_pos]=slice_policyX
			policyX[g_pos]=slice_policyY

		child_policy1 = policyY
		child_policy2 = policyX
		# IMPLEMENT!
		# generate a child policy from cross-over of the parents

		return child_policy1, child_policy2

	def mutation (self, policy, nun_gen_2_mutate = 2, mut_prob_thr=0.05):

		mutation_prob = float(np.random.choice(100,1))/100
		available_positions = np.arange(0,5)

		if mutation_prob <= mut_prob_thr:

			positions = []
			mutations = []

			for _ in range(nun_gen_2_mutate):

				position = int( np.random.choice( 6, 1) ) ## 5 gen positions
				if position == 0:
					all_moves = self.tt_gen_pos_01 - 1
				elif position == 1:
					all_moves = self.tt_gen_pos_02 - 1
				elif position == 2:
					all_moves = self.tt_gen_pos_03 - 1
				elif position == 3:
					all_moves = self.tt_gen_pos_04 - 1
				elif position == 4:
					all_moves = self.tt_gen_pos_05 - 1
				elif position == 5:
					all_moves = self.tt_gen_pos_06 - 1
				if all_moves == 0:
					move =0
				else:
					move = int( np.random.choice( all_moves, 1 ) )    

				if position not in positions:
					pos_mask = available_positions != position
					available_positions = available_positions[pos_mask]
					positions.append(position)
				else:
					position = int(np.random.choice(available_positions,1))
					positions.append(position)
				
				mutate = [move, position]  
				mutations.append(mutate)
				policy[position] = move

		return policy
	
	def update_agents(self):

		self.agent_selection()  ### this updates which agents will perfom crossover in the self.parents where parents = [ [policy1,policy2], [policyN,policyM]]
		
		new_policies = []
		for policy1, policy2 in self.parents:
			child_policy1, child_policy2 = self.crossover(policy1, policy2)

			child_policy1 = self.mutation(child_policy1)
			child_policy2 = self.mutation(child_policy2)

			new_policies.append(child_policy1)
			new_policies.append(child_policy2)

		### drop_weaker older agents
		new_agents = int(len(new_policies))
		for _ in range (new_agents):
			self.overall_results.pop(-1)

		### add new agents
		print("\n Add New Policies ")
		count =0
		for new_policy in  new_policies:
			new_policy_ls = new_policy.tolist() ### return from numpy to list format
			new_result = self.generate_single_agent(new_policy_ls)
			#print("new_policy = ", count, new_policy)
			#print("overall_results lenght= ", len(self.overall_results))
			#print("shape overall_results = ", np.array(self.overall_results).shape)
			self.overall_results.append(new_result)
			self.store_policy_result(new_result)
			count += 1
		
	def show_result(self,result,num):

		policy = result[0]

		lr_index = policy[0]; lr = self.lr_ls[lr_index]
		decay_index = policy[1]; decay = self.decay_ls[decay_index] 
		bs_index = policy[2]; batch_size = self.batch_size_ls[bs_index]
		ep_index = policy[3]; epochs=self.epoch_ls[ep_index]
		dr_index = policy[4]; dropout_rate = self.dropout_ls[dr_index] 
		opt_index = policy[5]; optimizer = self.optimizer_ls[opt_index]            
		
		history = result[1]
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		title = 'model number {} time to train= {} seconds and accuracy {} % \n'.format( num, np.round(result[2], decimals=2), np.round(result[3], decimals = 3)*100)
		title += ' - optimizer: {} lr= {} decay= {} \n'.format(optimizer, lr, decay)
		title += ' - neurons : dropout_rate {} \n'.format(dropout_rate)
		title += ' - batch_size= {} epochs= {} '.format(batch_size, epochs)
				
		plt.title( title)
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		plt.ylim((0,15))
		plt.show()

	def best_models(self, final_models = 3, max_generations = 5):

		count = 0
		while count <= max_generations:
			print("#"*200)
			print("#"*25, " Generation {}".format(count), "#"*25,"\n")
			if count == 0:
				self.initialize_agents()
			else:
				self.update_agents()
			self.evaluate_policies()
			count += 1
		
		best_models = []
		for num in range(final_models):
			result = self.overall_results[num]
			self.show_result(result,num)
			best_models.append(result)
		return result

print("#"*25, " Set hyperparameters lists", "#"*25,"\n")
lr_ls = [1e-3,1e-4]
decay_ls = [1e-6,1e-7]
batch_size_ls = [32,500]
epoch_ls = [150,250]
dropout_ls = [0.05, 0.01]
optimizer_ls = ["SGD","AdaGrad","RMSProp"]

print("#"*25, " Genetic Algorithm Start", "#"*25,"\n")
ga = Genetic_Algorithm(lr_ls,decay_ls,batch_size_ls, epoch_ls, dropout_ls, optimizer_ls,
						X_train, y_oh_train, X_test, y_oh_test, y_test, num_agents=30)

print("#"*25, " Set hyperparameters lists", "#"*25,"\n")
best = ga.best_models(final_models = 10, max_generations = 2)

all_results = ga.tested_policies

for policy_name in all_results.keys():
	policy = ga.convert_name_2_policy(policy_name)

	lr_index = policy[0]
	decay_index = policy[1]
	bs_index = policy[2]
	ep_index = policy[3]
	dr_index = policy[4]
	opt_index = policy[5]
	
	dropout_rate = ga.dropout_ls[dr_index],          
	lr = ga.lr_ls[lr_index], 
	decay = ga.decay_ls[decay_index],           
	batch_size = ga.batch_size_ls[bs_index], 
	epochs=ga.epoch_ls[ep_index]
	optimizer = ga.optimizer_ls[opt_index]

	for result in all_results[policy_name]:
		
		title = '\n model - time to train= {} seconds and accuracy {} % \n'.format( np.round(result[2], decimals=2), np.round(result[3], decimals = 3)*100)
		title += ' - optimizer :{} lr= {} decay= {} \n'.format(optimizer, lr, decay)
		title += ' - neurons : dropout_rate {} \n'.format(dropout_rate)
		title += ' - batch_size= {} epochs= {} '.format(batch_size, epochs)
		print(title)


# %%


# ### Question 1
# **The data set**
# 
# Plot a three examples from the data set.
# * What type of data are in the data set?
# 
#     <span style="color:red"> 
#       The MNIST dataset contains grayscale images of handwritten numbers from zero to nine. 
# 		When they are imported in the fetch_openml an array with shape images vs 784 is retrieved. 
# 		784 is the flattened form of the image and the input values are pixel values ranging from 0 – 255. 
# 		Because it has only one color channel, it means that the dataset is grayscale.
#      </span>
#     
# * What does the line ```X = X.reshape(X.shape[0], 28, 28, 1)``` do?
#     <span style="color:red"> 
#       This operation reshapre the flatten array into a new array with (columns, height, witdh, colour_channel)
#      </span>
# 
# Look at how the encoding of the targets (i.e. ```y```) is changed. E.g. the lines
# ```
#     y_oh = np.zeros((num_tot, num_classes))
#     y_oh[range(num_tot), y] = 1
# ```
# Print out a few rows of ```y``` next to ```y_oh```.
# * What is the relationship between ```y``` and ```y_oh```?
#   
#     <span style="color:red"> 
#      "y" means the supervised information that tells you the specific input must belong to a output
#      "y" ranges from 0 to 9 while y_oh is the one_hot encoding.
#       </span>
#     
#     
# * What is the type of encoding in ```y_oh``` called and why is it used?
# 
#     <span style="color:red"> 
#           y_oh is the one_hot encoding and its used for classification problems.
#           Since deeplearning are based on math and numbers, 
#           the output of a classification must be a numerical value and not a string. 
#           Therefore a multiclassification problem should have a one hot encoded 
#           so that the ouput of the neural network can be numerically compared with the prediction.
#           
#           Ex: A NN with three classes ("dog,cat,human") should have 3 columns one for dog, one for cat and another for humam
#           a dog picture would have its one hot encoding  (1 , 0 , 0) because it belongs to one class 
#           a cat picture would have its one hot encoding  (0 , 0 , 1) and a human would have (0, 0, 1)
#                             </span>
#     
#     
# * Plot three data examples in the same figure and set the correct label as title. 
#     * It should be possible to see what the data represent.

# %%
# ### Question 2
# **The model**
# 
# Below is some code for bulding and training a model with Keras.
# * What type of network is implemented below? I.e. a normal MLP, RNN, CNN, Logistic Regression...?
#     <span style="color:red"> The type of the network used is CNN (Convolutional Neural Networks) </span> 
# * What does ```Dropout()``` do?
#     <span style="color:red"> "Dropout randomly disconnects neurons connections to make the CNN more genralist, 
#                               and thus reducing the probability of overfitting" </span>
# * Which type of activation function is used for the hidden layers?
#     <span style="color:red"> Rectified Linear Unit (ReLU) </span>
# * Which type of activation function is used for the output layer?
#     <span style="color:red"> Softmax </span>
# * Why are two different activation functions used?
#     <span style="color:red"> ReLU is used to solve the vanishing gradient problem 
#							   and it reduced the influence of negative values after a convolution,
#                              while softmax is used to transform the output layer probabilities
#                              into the most probable output </span>
# * What optimizer is used in the model below?
#     <span style="color:red"> Although the name of the function is SGD (Stochastic Gradient Descent)
# 						the model useds batch gradient descent </span>
# * How often are the weights updated (i.e. after how many data examples)?
#     <span style="color:red"> The epoch and batches will define when the weights are updated.   
# 							Since this specific model is using mini-batch gradient descent,
#  							the model will update their weights after completing a batch.
#  							A dataset with 800 images with a batch size of 32 
# 							will have 25 weights updates per epoch. </span>
# * What loss function is  used?
# 
#     <span style="color:red"> Categorical crossentropy </span>
# 
# 
# * How many parameters (i.e. weights and biases, NOT hyper-parameters) does the model have?
# 
#     <span style="color:red"> <*answer here*> </span>
# 
#from IPython import embed; embed()

# %%
# ### Question 3
# 
# * **Vizualize the training**. Use the model above to observe the training process. Train it for 150 epochs and then plot both "loss" and "val_loss" (i.e. loss on the valiadtion set, here the terms "validation set" and "test set" are used interchangably, but this is not always true). What is the optimal number of epochs for minimizing the test set loss? 
#     * Remember to first reset the weights (```model.reset_states()```), otherwise the training just continues from where it was stopped earlier.
# 
# * **Optimizer**. Select three different optimizers and for each find the close-to-optimal hyper-parameter(s). In your answer, include a) your three choises, b) best hyper-parameters for each of the three optimizers and, c) the code that produced the results.
#     * NOTEa that how long the training takes varies with optimizer. I.e., make sure that the model is trained for long enough to reach optimal performance.
# 
# * **Dropout**. Use the best optimizer and do hyper-parameter seach and find the best value for ```Dropout()```.
# 
# * **Best model**. Combine the what you learned from the above three questions to build the best model. How much better is it than the worst and average models?
# 
#     <span style="color:red"> <*answer here*> </span>
# 
# 
# * **Results on the test set**. When doing this search for good model configuration/hyper-parameter values, the data set was split into *two* parts: a training set and a test set (the term "validation" was used interchangably wiht "test"). For your final model, is the performance (i.e. accuracy) on the test set representative for the performance one would expect on a previously unseen data set (drawn from the same distribution)? Why?
# 
#     <span style="color:red"> <*answer here*> </span>
# 
# 
# ## Further information
# For ideas about hyper-parameter tuning, take a look at the strategies described in the sklearn documentation under [model selection](https://scikit-learn.org/stable/model_selection.html), or in this [blog post](https://blog.tensorflow.org/2020/01/hyperparameter-tuning-with-keras-tuner.html) from TensorFlow. For a more thorough discussion about optimizers see [this video](https://www.youtube.com/watch?v=DiNzQP7kK-s) discussing the article [Descending through a Crowded Valley -- Benchmarking Deep Learning Optimizers](https://arxiv.org/abs/2007.01547).
# 
# 
# **Good luck!**


# %%
