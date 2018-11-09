import agent_double as agent_double
import agent_onehot as agent_one
import agent_one_tseFeat as agent_one_tseFeat
import Backgammon
import numpy as np
from tqdm import tqdm
import pickle
# train

# n = 10
# net = agent.policy_nn()
# val_nn = agent.val_func_nn()
#agent = agent_one.net()
#agent = agent_double.net()
#agent = agent_one_tseFeat.net()
agent = pickle.load(open('saved_net_onehot', 'rb'))


def main():
    winners = {}
    winners["1"] = 0
    winners["-1"] = 0  # Collecting stats of the games
    nGames = 10000    # how many games?
    arr = np.zeros(nGames)
    for g in tqdm(range(nGames)):
        
#        w=new_agent.actor.theta
#        print(w)
        
        ###Zero eligibility traces (according to psudo code)
        agent.actor.zero_el()
        agent.critic.zero_el()
        
        winner = Backgammon.play_a_game(commentary=False, net=agent)
        
        
        
        winners[str(winner)] += 1
        arr[g] = winner
#        if(g % 100 == 0):
#            print(new_agent.torch_nn_policy.theta)
    # print(winners)
#    
    ##Save the agent
    file_net = open('saved_net_one_2', 'wb')
    pickle.dump(agent, file_net)
    file_net.close()
    
    print("Out of", nGames, "games,")
    print("player", 1, "won", winners["1"], "times and")
    print("player", -1, "won", winners["-1"], "times")


if __name__ == '__main__':
    main()