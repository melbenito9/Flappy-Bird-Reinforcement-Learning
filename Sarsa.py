import os
import sys
import random
import json
import numpy as np
import gym
import pickle
from FBEnv import FBSarsa
from Sample import FBSample
sys.path.append('./SRC')

#SARSA Agent and Initialization
class Sarsa(FBSample):
   
    def __init__(self, Act, Flap_prob=0.5, Rounding=None):
        super().__init__(Act)
        self.Flap_prob = Flap_prob
        self.Q_Values = defaultdict(float)
        self.env = FBSarsa(gym.make('FlappyBird-v0'), Rounding=Rounding)

    def Actions(self, State):
        def randomAction():
            if random.random() < self.Flap_prob:
                return 0
            return 1
        
        if random.random() < self.Epsilon:
            return randomAction()

        Q_Values = [self.Q_Values.get((State, action), 0) for action in self.Act]

        if Q_Values[0] < Q_Values[1]:
            return 1
        elif Q_Values[0] > Q_Values[1]:
            return 0
        else:
            return randomAction()

    #Saves the Q-values
    def saveQValues(self):
        Q_Values_Save = {key[0] + ' action ' + str(key[1]): self.Q_Values[key] for key in self.Q_Values}
        with open('Q_Values.json', 'w') as fp:
            json.dump(Q_Values_Save, fp)
    
    #Loads the Q-values
    def Load_Q_Values(self):
        def parseKey(key):
            state = key[:-9]
            action = int(key[-1])
            return (state, action)

        with open('Q_Values.json') as fp:
            Load_Q = json.load(fp)
            self.Q_Values = {parseKey(key): Load_Q[key] for key in Load_Q}

    #To train the agent
    def train(self, Order = 'forward', Iterations = 10000, Epsilon = 0.1, Gamma = 1,
              ETA = 0.9, Epsilon_Decay = False, ETA_Decay = False, Evaluation_Freq = 500,
              Iters_Eval = 1000):
    
        self.Epsilon = Epsilon
        self.InitialEpsilon = Epsilon
        self.Gamma = Gamma
        self.ETA = ETA
        self.Epsilon_Decay = Epsilon_Decay
        self.ETA_Decay = ETA_Decay
        self.Evaluation_Freq = Evaluation_Freq
        self.Iters_Eval = Iters_Eval
        self.env.seed(random.randint(0, 100))

        done = False
        Max_Score = 0
        Max_Reward = 0

        main_dict ={"non_colide":0,"pipe":0,"dead":0}
        print("TRAINING SARSA ON {} ITERATIONS".format(Iterations))
        # import pdb;pdb.set_trace()
        for i in range(Iterations):
            temp_list=[]
            temp ={"non_colide":0,"pipe":0,"dead":0}
            if i % 50 == 0 or i == Iterations - 1:
                print("Iter: ", i)
            # import pdb;pdb.set_trace()
           
            self.Epsilon = self.InitialEpsilon / (i + 1) if self.Epsilon_Decay \
                           else self.InitialEpsilon
            score = 0
            Total_Reward = 0
            ob = self.env.reset()
            gameIter = []
            State = self.env.getGameState()
            action = self.Actions(State)
           
            while True:
                nextState, reward, done, _ = self.env.step(action)
                if reward==5:
                    temp['pipe']= temp['pipe']+1
                    # print(temp)
                elif reward==-100:
                    temp['dead']= temp['dead']+1
                    # print(temp)
                elif reward==0.1:
                    temp['non_colide']= temp['non_colide']+1
                else:
                    pass
                print(temp)
                temp_list.append(temp)


                # import pdb;pdb.set_trace()
                nextAction = self.Actions(nextState)
                gameIter.append((State, action, reward, nextState, nextAction))
                State = nextState
                action = nextAction
                Total_Reward += reward
                if reward >= 1:
                    score += 1
                if done:
                    break
            import json
            with open("train/train_dict"+str(i)+".json","w") as f1:json.dump(temp_list,f1)
           
           
            if score > Max_Score: Max_Score = score
            if Total_Reward > Max_Reward: Max_Reward = Total_Reward
           
            if Order == 'forward':
                for (State, action, reward, nextState, nextAction) in gameIter:
                    self.updateQ(State, action, reward, nextState, nextAction)
            else:
                for (State, action, reward, nextState, nextAction) in gameIter[::-1]:
                    self.updateQ(State, action, reward, nextState, nextAction)
            # import pdb;pdb.set_trace()
            with open("q_values/q_values_"+str(i)+".pkl","wb") as f1:pickle.dump(self.Q_Values,f1)
            # import pdb;pdb.set_trace()
            #Q_Values_Save = {key[0] + ' action ' + str(key[1]): self.Q_Values[key] for key in self.Q_Values}
               
            if self.ETA_Decay:
                self.ETA *= (i + 1) / (i + 2)
           
            # import pdb;pdb.set_trace()
            if (i + 1) % self.Evaluation_Freq == 0:
                print("*********************OK SAVING Q VALUES*****************************")

                output = self.test(Iterations = self.Iters_Eval)
                self.saveOutput(output, i + 1)
                # import pdb;pdb.set_trace()
                self.saveQValues()
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        self.env.close()
        print("Max Train Score : ", Max_Score)
        print("Max Train Reward : ", Max_Reward)
        # import json
        # with open("train/train_dict.json","w") as f1:json.dump(main_dict,f1)
        print()
   
    agent.saveQValues()
    agent.env.close()

    print("Max Test Score : ", Max_Score)
    print("Max Test Reward: ", Max_Reward)
    print()```

    def test(self, Iterations=10000):
        
    self.Epsilon = 0
    self.env.seed(0)

    done = False
    Max_Score = 0
    Max_Reward = 0
    output = defaultdict(int)
    main_dict = {"non_colide": 0, "pipe": 0, "dead": 0}

    for i in range(Iterations):
        temp = {"non_colide": 0, "pipe": 0, "dead": 0}
        temp_list = []
        score = 0
        Total_Reward = 0
        ob = self.env.reset()
        State = self.env.getGameState()
           
        while True:
            action = self.Actions(State)
            State, reward, done, _ = self.env.step(action)
            if reward == 5:
                temp['pipe'] = temp['pipe'] + 1
                print(temp)
            elif reward == -100:
                temp['dead'] = temp['dead'] + 1
                print(temp)
            elif reward == 0.1:
                temp['non_colide'] = temp['non_colide'] + 1
            else:
                pass
            print(temp)
            temp_list.append(temp.copy())
#            self.env.render()  # Uncomment it to display graphics.
            Total_Reward += reward
            if reward >= 1:
                score += 1
            if done:
                break
        
        with open(f"test/test_dict{i}.json", "w") as f:
            json.dump(temp_list, f)

        output[score] += 1
        if score > Max_Score:
            Max_Score = score
        if Total_Reward > Max_Reward:
            Max_Reward = Total_Reward
   
    self.env.close()
    print("Maximum Test Score: ", Max_Score)
    print("Maximum Test Reward: ", Max_Reward)
    print()
    with open("test/test.json", "w") as f:
        json.dump(main_dict, f)
    return output

def updateQ(self, State, action, reward, nextState, nextAction):
    oldQValue = self.Q_Values.get((State, action), 0)
    nextQValue = self.Q_Values.get((nextState, nextAction), 0)
    newQValue = oldQValue + self.ETA * (reward + self.discount * nextQValue - oldQValue)
    self.Q_Values[(state, action)] = newQValue
