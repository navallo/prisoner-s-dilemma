import numpy as np
import random
import matplotlib.pyplot as plt


class DFA_4():
	def __init__(self, genome, id):
		self.genome = genome
		self.init_state = int((self.genome[0:2]), 2)
		self.action_dict = {
			0: int(self.genome[2]),
			1: int(self.genome[7]),
			2: int(self.genome[12]),
			3: int(self.genome[17])
		}
		self.trans_dict = {
			(0, 0): int(self.genome[3:5], 2),
			(0, 1): int(self.genome[5:7], 2),
			(1, 0): int(self.genome[8:10], 2),
			(1, 1): int(self.genome[10:12], 2),
			(2, 0): int(self.genome[13:15], 2),
			(2, 1): int(self.genome[15:17], 2),
			(3, 0): int(self.genome[18:20], 2),
			(3, 1): int(self.genome[20:22], 2)
		}
		self.state = self.init_state
		self.reward = 0
		self.id = id

	def transfer(self, action):
		new_state = self.trans_dict[(self.state, action)]
		self.state = new_state

	def act(self):
		return self.action_dict[self.state]


class PD():
	def __init__(self, num):
		self.num = num
		self.agents = {}
		for i in range(self.num):
			genome = '00'
			for gene in np.random.randint(2, size=20).tolist():
				genome += str(gene)
			self.agents['dfa_' + str(i)] = DFA_4(genome=genome, id=i)
		self.reward_dict = {
			(0, 0): (3, 3),
			(0, 1): (0, 5),
			(1, 0): (5, 0),
			(1, 1): (1, 1)
		}
		self.action_count = {
			(0, 0): 0,
			(0, 1): 0,
			(1, 0): 0,
			(1, 1): 0
		}
		self.action_his = []

	def show_agents(self):
		for i in range(self.num):
			print('id', self.agents['dfa_' + str(i)].id, 'reward',
				  self.agents['dfa_' + str(i)].reward)
			print('genome', self.agents['dfa_' + str(i)].genome)
			print('action_dict',self.agents['dfa_' + str(i)].action_dict)
			print('trans_dict',self.agents['dfa_' + str(i)].trans_dict)


	def compete(self, trails, repeat):
		list_1 = np.random.randint(self.num, size=trails)
		list_2 = np.random.randint(self.num, size=trails)
		for i in range(trails):

			# print('--------------------')
			# print('match',list_1[i],list_2[i])
			if list_1[i] != list_2[i]:
				agent_a = self.agents['dfa_' + str(list_1[i])]
				agent_b = self.agents['dfa_' + str(list_2[i])]

				for j in range(repeat):
					# print('state',agent_a.state,agent_b.state)
					action_a = agent_a.act()
					action_b = agent_b.act()
					self.action_count[(action_a, action_b)] += 1
					# print('action',action_a,action_b)
					reward = self.reward_dict[(action_a, action_b)]
					# print('reward',reward)
					agent_a.reward += reward[0]
					agent_b.reward += reward[1]

					agent_a.transfer(action_b)
					agent_b.transfer(action_a)
					# print('new_state',agent_a.state,agent_b.state)
					# print('all_reward',agent_a.reward,agent_b.reward)
			else:
				pass

	def evolve(self, elite_rate, mutation_rate):
		elite_num = int(elite_rate*self.num)
		fitness = []
		for i in range(self.num):
			fitness.append(self.agents['dfa_' + str(i)].reward)
		index = np.array(fitness).argsort().tolist()

		deads = index[:self.num - elite_num]
		elites = index[self.num - elite_num:]

		# print('fitness',fitness)
		# print('index',index)
		# print('deads',deads)
		# print('elites',elites)

		for dead_agent in deads:
			father_id = random.choice(elites)
			father_genome = self.agents['dfa_' + str(father_id)].genome

			tmp=list(father_genome)
			for i in range(2,22):
				if random.random() <= mutation_rate:
					if tmp[i] == '0':
						tmp[i] = '1'
					elif tmp[i] == '1':
						tmp[i] = '0'
					else:
						raise Exception("ERROR!!")
			father_genome=''.join(tmp)

			self.agents['dfa_' + str(dead_agent)] = DFA_4(genome=father_genome, id=dead_agent)

		for i in range(self.num):
			self.agents['dfa_' + str(i)].reward = 0
			self.agents['dfa_' + str(i)].state = self.agents['dfa_' + str(i)].init_state

		all_actions = 0
		for value in self.action_count.values():
			all_actions += value
		two_cooperate = self.action_count[(0, 0)]/all_actions
		one_defect = (self.action_count[(0, 1)] + self.action_count[(1, 0)])/all_actions
		two_defect = self.action_count[(1, 1)]/all_actions

		self.action_his.append([two_cooperate,one_defect,two_defect])
		self.action_count = {
			(0, 0): 0,
			(0, 1): 0,
			(1, 0): 0,
			(1, 1): 0
		}

def main():
	EPOCH = 1000
	pd = PD(num=50)
	# pd.show_agents()
	for count in range(EPOCH):
		# print('----------------------- EPOCH',count,'-----------------------')
		pd.compete(trails=100, repeat = 20)
		# print('action_count',pd.action_count)
		pd.evolve(elite_rate = 0.1, mutation_rate=0.005)
	pd.show_agents()
	# print('history',pd.action_his)
	history = np.array(pd.action_his)
	# print(history)
	plt.plot(history)
	plt.title('prisoner dilemma')
	plt.xlabel('Generations')
	plt.ylabel('rate of each joint-actions')
	plt.legend(['two cooperate','one defect','two defect'], loc='upper right')
	plt.show()

if __name__ == "__main__":
	main()
