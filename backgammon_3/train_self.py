import agent_double as agent_double
import agent_onehot as agent_one
import agent_one_tseFeat as agent_one_tseFeat
import agent_share as agent_share
import agent_share_2 as agent_share2
import agent_final as agent_final
import agent_ts as agent_ts
import Backgammon_self
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# train

# n = 10
# net = agent.policy_nn()
# val_nn = agent.val_func_nn()
#agent = agent_one.net()
#agent = agent_share2.net()
#agent =agent_share.net()
#agent =agent_final.net()
agent = agent_ts.net()
#agent = agent_double.net()
#agent = agent_one_tseFeat.net()
#agent = pickle.load(open('saved_net_onehot', 'rb'))


def main():
    winners = {}
    winners["1"] = 0
    winners["-1"] = 0  # Collecting stats of the games
    nGames = 10001   # how many games?
    numWins=[]
    for g in tqdm(range(nGames)):

        if g==5000:
            agent.alpha1/=10
            agent.alpha2/=10
            agent.alphaC/=10
            agent.alphaA/=10
            
        ###Compete against a random player and record performance
        if (g%1000==0):
            winners = {}
            winners["1"] = 0
            winners["-1"] = 0
            
            for i in range(100):
#                agent.actor.zero_el()
#                agent.critic.zero_el()
                agent.reset_old_boards()
                agent.zero_el()
                agent.firstMove=True
        
                winner = Backgammon_self.play_a_game(commentary=False, net=agent, random=True, learn=False)
        
                winners[str(winner)] += 1
                
            numWins.append(winners["1"])
            
            
            
            
        ###Zero eligibility traces (according to psudo code)
#        agent.actor.zero_el()
#        agent.critic.zero_el()
        agent.reset_old_boards()
        agent.zero_el()
        agent.firstMove=True
        
        winner = Backgammon_self.play_a_game(commentary=False, net=agent)
        
        
        
#        winners[str(winner)] += 1
    
    ##Save the agent
    file_net = open('saved_agent_ts', 'wb')
    pickle.dump(agent, file_net)
    file_net.close()
    
#    print("Out of", nGames, "games,")
#    print("player", 1, "won", winners["1"], "times and")
#    print("player", -1, "won", winners["-1"], "times")
    
    ###Plot the performancse against a random player
    x=np.arange(0, nGames, 1000)
    fig = plt.figure()
    #plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(111)
    ax.plot(x, numWins)
    
    
    
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()