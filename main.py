#main file of the SARSA program
 
import os, sys
import warnings
import argparse
from FBEnv import FBEnv
from FBUtils import FBUtils
from Sarsa import Sarsa
import warnings

sys.path.append('.')
sys.path.append('./AG')
warnings.filterwarnings('ignore')


agents = ['SARSA']
orders = ['forward', 'backward']


def parseArgs():
    parser = argparse.ArgumentParser(description = 'An AI Agent for Flappy Bird.',
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', type = str, default = 'SARSA',
                        help = 'SARSA algorithm.', choices = agents)
    
    # Parameters for SARSA
    parser.add_argument('--Rounding', type = int, default = None,
                        help = 'Level of discretization.')
    parser.add_argument('--Flap_Prob', type = float, default = 0.1,
                        help = 'Probability of flapping in epsilon-greedy policy.')
    parser.add_argument('--Order', type = str, default = 'forward',
                        choices = orders, help = 'Order of Q-value updates.')
    parser.add_argument('--Gamma', type = float, default = 1.,
                        help = 'Discount factor.')
    parser.add_argument('--Train_Iterations', type = int, default = 1000,
                        help = 'Number of training iterations.')
    parser.add_argument('--Test_Iteration', type = int, default = 500,
                        help = 'Number of testing iterations.')
    parser.add_argument('--Evaluation_Freq', type = int, default = 250,
                        help = 'Frequency of running evaluation.')
    parser.add_argument('--Epsilon', type = float, default = 0.,
                        help = 'Epsilon-Greedy policy')
    parser.add_argument('--Alpha', type = float, default = 0.9,
                        help = 'Learning rate.')
    parser.add_argument('--Epsilon_Decay', action = 'store_true',
                        help = 'Use epsilon decay or not.')
    parser.add_argument('--LR_Decay', action = 'store_true',
                        help = 'Use learning rate decay or not.')
    
    args = parser.parse_known_args()[0]
    return args

    
def main():
    print("Hello! Let's play Flappy Bird.")
    args = parseArgs()
    
    if args.model == 'SARSA':
        agent = Sarsa(Act = [0, 1], Rounding = args.Rounding, Flap_Prob = args.Flap_Prob)
        agent.train(Order = args.Order, Iterations = args.Train_Iterations, Epsilon = args.Epsilon,
                    Gamma = args.Gamma, ETA = args.Alpha, Epsilon_Decay = args.Epsilon_Decay,
                    ETA_Decay = args.LR_Decay, Evaluation_Freq = args.Evaluation_Freq,
                    Iters_Eval= args.Test_Iterations)
        agent.saveQValues()
        
        
if __name__ == '__main__':
    main()
