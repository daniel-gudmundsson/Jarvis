#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""
import numpy as np
import Backgammon
import torch
from torch.autograd import Variable
from numpy.random import choice

#An object that contains two neural netoworks (one critic, one actor)
class net():
    def __init__(self):
        self.critic=critic() #The critic nnn
        self.actor=actor() #the actor nnn
        self.gamma=1
        self.delta=0
    



class critic():
    ###Initialized the critic
    def __init__(self):
        self.device = torch.device('cpu')
        ###Weights and bias
        self.w1 = Variable(torch.randn(99, 198, device=self.device , dtype=torch.float), requires_grad=True)
        self.b1 = Variable(torch.zeros((99, 1), device=self.device , dtype=torch.float), requires_grad=True)
#        self.w2 = Variable(torch.randn(198, 99, device=self.device , dtype=torch.float), requires_grad=True)
#        self.b2 = Variable(torch.zeros((198, 1), device=self.device , dtype=torch.float), requires_grad=True) ##Use only one hidden layer to start with
        ###Eligibility traces
        self.Z_w1 = torch.zeros(self.w1.size(), device=self.device, dtype=torch.float)
        self.Z_b1 = torch.zeros(self.b1.size(), device=self.device, dtype=torch.float)
#        self.Z_w2 = torch.zeros(self.w2.size(), device=self.device, dtype=torch.float)
#        self.Z_b2 = torch.zeros(self.b2.size(), device=self.device, dtype=torch.float)
        
        ###Final step
        #self.W=Variable(torch.zeros((1, 198), device=self.device , dtype=torch.float), requires_grad=True)
        self.W=Variable(torch.randn((1, 99), device=self.device , dtype=torch.float), requires_grad=True)
        self.B=Variable(torch.zeros((1, 1), device=self.device , dtype=torch.float), requires_grad=True)
        self.Z_W = torch.zeros(self.W.size(), device=self.device, dtype=torch.float)
        self.Z_B = torch.zeros(self.B.size(), device=self.device, dtype=torch.float)
        
        self.y_sigmoid = 0
        self.target=0
        self.oldtarget=0
        
        ###Step sizes for each layer
        self.alpha1 = 0.01
        self.alpha2 = 0.01
        self.alpha3 = 0.01
        
        self.lam=1
        
    ###Forward propgation for both the old board and the new board    
    def forward(self, xnew, xold):
        ###First forward the new board
        x_prime = Variable(torch.tensor(xnew, dtype=torch.float, device=self.device)).view(198, 1)
        
        ###First layer
        x1=torch.mm(self.w1, x_prime) + self.b1
        x1_sigmoid=x1.sigmoid()
        ###Second layer (start with only one for now)
#        x2=torch.mm(self.w2, x1_sigmoid) + self.b2
#        x2_sigmoid=x2.sigmoid()
        ###Final step
        y=torch.mm(self.W, x1_sigmoid) + self.B
        self.y_sigmoid=y.sigmoid()
        self.target = self.y_sigmoid.detach().cpu().numpy()
        
        ###Do the same for the old board
        x_prime = Variable(torch.tensor(xold, dtype=torch.float, device=self.device)).view(198, 1)
        
        ###First layer
        x1=torch.mm(self.w1, x_prime) + self.b1
        x1_sigmoid=x1.sigmoid()
        ###Second layer (start with only one for now)
#        x2=torch.mm(self.w2, x1_sigmoid) + self.b2
#        x2_sigmoid=x2.sigmoid()
        ###Final step
        y=torch.mm(self.W, x1_sigmoid) + self.B
        self.y_sigmoid=y.sigmoid()
        self.oldtarget = self.y_sigmoid.detach().cpu().numpy()
        
        return self.target, self.oldtarget
    
    def backward(self,R, delta, gamma):
        ###Calculate the gradiants
        self.y_sigmoid.backward()
        
        ###Update eligibility traces
#        self.Z_w2 = gamma * lam * self.Z_w2 + self.w2.grad.data
#        self.Z_b2 = gamma * lam * self.Z_b2 + self.b2.grad.data
        ###Only using one hidden layer for now
        self.Z_w1 = gamma * self.lam * self.Z_w1 + self.w1.grad.data
        self.Z_b1 = gamma * self.lam * self.Z_b1 + self.b1.grad.data
        
        self.Z_W = gamma * self.lam * self.Z_W + self.W.grad.data
        self.Z_B = gamma * self.lam * self.Z_B + self.B.grad.data
        
        ###Zero the gradiants
#        self.w2.grad.data.zero_()
#        self.b2.grad.data.zero_()
        self.w1.grad.data.zero_()
        self.b1.grad.data.zero_()
        self.W.grad.data.zero_()
        self.B.grad.data.zero_()
        
        #self.delta=R+self.gamma*self.target-self.oldtarget
        
        self.w1.data = self.w1.data + self.alpha1 * delta * self.Z_w1
        self.b1.data = self.b1.data + self.alpha1 * delta * self.Z_b1
#        self.w2.data = self.w2.data + self.alpha2 * self.delta * self.Z_w2
#        self.b2.data = self.b2.data + self.alpha2 *self.delta * self.Z_b2
        
        self.W.data = self.W.data + self.alpha3 * delta * self.Z_W
        self.B.data = self.B.data + self.alpha3 * delta * self.Z_B
        

    
 ### A class for the actor   
class actor():
    ### Initialized the actor
    def __init__(self):
        self.device = torch.device('cpu')
        ###Weights and bias
        self.w1 = Variable(torch.randn(99, 198, device=self.device , dtype=torch.float), requires_grad=True)
        self.b1 = Variable(torch.zeros((99, 1), device=self.device , dtype=torch.float), requires_grad=True)
#        self.w2 = Variable(torch.randn(198, 99, device=self.device , dtype=torch.float), requires_grad=True)
#        self.b2 = Variable(torch.zeros((198, 1), device=self.device , dtype=torch.float), requires_grad=True) ##Use only one hidden layer to start with
        ###Eligibility traces
        self.Z_w1 = torch.zeros(self.w1.size(), device=self.device, dtype=torch.float)
        self.Z_b1 = torch.zeros(self.b1.size(), device=self.device, dtype=torch.float)
#        self.Z_w2 = torch.zeros(self.w2.size(), device=self.device, dtype=torch.float)
#        self.Z_b2 = torch.zeros(self.b2.size(), device=self.device, dtype=torch.float)
        
        ###Final step
        #self.W=Variable(torch.zeros((1, 198), device=self.device , dtype=torch.float), requires_grad=True)
        self.theta=Variable(torch.randn((1, 99), device=self.device , dtype=torch.float), requires_grad=True)
        self.B=Variable(torch.zeros((1, 1), device=self.device , dtype=torch.float), requires_grad=True)
        self.Z_theta = torch.zeros(self.theta.size(), device=self.device, dtype=torch.float)
        self.Z_B = torch.zeros(self.B.size(), device=self.device, dtype=torch.float)
        
#        self.y_sigmoid = 0
#        self.target=0
#        self.oldtarget=0
        
        self.softmax=0
        self.ln_softmax=0 ###Not sure about this
        
        ###Step sizes for each layer
        self.alpha1 = 0.01
        self.alpha2 = 0.01
        self.alpha3 = 0.01
        
        self.lam=1
    
    ### Forward propgation
    ### Returns probabilities of each action and also the logarithm which will be used for the backprobogation
    def forward(self, X):
        values=torch.zeros(len(X))
        c=0
        ###First feed forward each value with the final step theta * new features (similar to the softmax example on page 322)
        for x in X:
            ###First step is similar the critic forward propogation
            x_prime = Variable(torch.tensor(x, dtype=torch.float, device=self.device)).view(198, 1)
            
                ###First layer
            x1=torch.mm(self.w1, x_prime) + self.b1
            x1_sigmoid=x1.sigmoid()
            ###Second layer (start with only one for now)
    #        x2=torch.mm(self.w2, x1_sigmoid) + self.b2
    #        x2_sigmoid=x2.sigmoid()
            ###Final step
            y=torch.mm(self.theta, x1_sigmoid) + self.B
            values[c]=y
            c+=1
        
        ###Now we have all the values and now we calculate the probabilities of each one using softmax
    
        probs=values.softmax(dim=0)
        log_probs=values.log_softmax(dim=0) ###This is later used for the backpropogation
        
        return probs, log_probs
    
    def backward(self, log_probs, delta, gamma):
        log_probs.backward() ### Calculate the gradiant
        
        ###Update eligibility traces
 #       self.Z_w2 = gamma * lam * self.Z_w2 + self.w2.grad.data
 #       self.Z_b2 = gamma * lam * self.Z_b2 + self.b2.grad.data
        ###Only using one hidden layer for now
        self.Z_w1 = gamma * self.lam * self.Z_w1 + self.w1.grad.data
        self.Z_b1 = gamma * self.lam * self.Z_b1 + self.b1.grad.data
        
        self.Z_theta= gamma * self.lam * self.Z_theta + self.theta.grad.data
        self.Z_B = gamma * self.lam * self.Z_B + self.B.grad.data
        
        ###Zero the gradiants
#        self.w2.grad.data.zero_()
#        self.b2.grad.data.zero_()
        self.w1.grad.data.zero_()
        self.b1.grad.data.zero_()
        self.theta.grad.data.zero_()
        self.B.grad.data.zero_()
        
        #self.delta=R+self.gamma*self.target-self.oldtarget
        
        self.w1.data = self.w1.data + self.alpha1 * delta * self.Z_w1
        self.b1.data = self.b1.data + self.alpha1 * delta * self.Z_b1
#        self.w2.data = self.w2.data + self.alpha2 * self.delta * self.Z_w2
#        self.b2.data = self.b2.data + self.alpha2 *self.delta * self.Z_b2
        
        self.theta.data = self.theta.data + self.alpha3 * delta * self.Z_theta
        self.B.data = self.B.data + self.alpha3 * delta * self.Z_B










def action(net, board_copy,dice,player,i):
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy
    
    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player)
    
    # if there are no moves available
    if len(possible_moves) == 0: 
        return [] 
    
    feature_boards=[]
    ###Create new features using Tsesauros
    for b in possible_boards:
        feature_boards.append(getFeatures(b, player))
    
    ### Get probabilites of each action via the actor forward
    probs, log_probs=net.actor.forward(feature_boards)
    
    ###index is used as a help to pick action
    index=np.arange(0, len(possible_boards))
    ###This works because numpy and pytorch hate each other
    probs=probs.detach().numpy()
    
    ###The index of the action chose
    i=choice(index, p=probs)
    move=possible_moves[i] ###Pick the next move according to the index selected
    newBoard=possible_boards[i] ###Pick the nex board according to the index selected
    newBoardFeatures=getFeatures(newBoard, player)
    
    R=0
    if(Backgammon.game_over(newBoard)): ###Did I win? If so the reward shall be +1
        R=1
    
    ### Now we update the neaural network
    
    target, oldtarget =net.critic.forward(newBoardFeatures,getFeatures(board_copy,player) )
    
    delta=R+net.gamma*target - oldtarget
    
    ###Update the critic via backpropgation
    net.critic.backward(R, delta, net.gamma)
    ###Update the actor via backpropogation
    net.actor.backward(log_probs[i], delta, net.gamma)
    
    
    return move
    
#    
#    
#    # make the best move according to the policy
#    
#    # policy missing, returns a random move for the time being
#    #
#    #
#    #
#    #
#    #
#    move = possible_moves[np.random.randint(len(possible_moves))]
#
#    return move

def getFeatures(board, player):
    features = np.zeros((198))
    for i in range(1, 24):
        board_val = board[i]
        place = (i - 1) * 4
        if(board_val < 0):
            place = place + 96
        # if(board_val == 0):
        #     features[place:place + 4] = 0
        if(abs(board_val) == 1):
            # print("one in place %i", place)
            features[place] = 1
            features[place + 1:place + 4] = 0
        if(abs(board_val) == 2):
            # print("two in place %i", place)
            features[place] = 1
            features[place + 1] = 1
            features[place + 2] = 0
            features[place + 3] = 0
        if(abs(board_val) == 3):
            # print("three in place %i", place)
            features[place] = 1
            features[place + 1] = 1
            features[place + 2] = 1
            features[place + 3] = 0
        if(abs(board_val) > 3):
            # print("more than three in place %i", place)
            features[place] = 1
            features[place + 1] = 1
            features[place + 2] = 1
            features[place + 3] = ((abs(board_val) - 3) / 2)
    features[192] = board[25] / 2
    features[193] = board[26] / 2
    features[194] = board[27] / 15
    features[195] = board[28] / 15
    if(player == 1):
        features[196] = 0
        features[197] = 0
    else:
        features[196] = 0
        features[197] = 0
    return features