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

class net():
    def __init__(self):
        self.device = torch.device('cpu')
        self.lam = 1
        self.w1 = Variable(torch.randn(99, 198, device=torch.device('cpu'), dtype=torch.float), requires_grad=True)
        self.b1 = Variable(torch.zeros((99, 1), device=torch.device('cpu'), dtype=torch.float), requires_grad=True)
        self.w2 = Variable(torch.randn(198, 99, device=torch.device('cpu'), dtype=torch.float), requires_grad=True)
        self.b2 = Variable(torch.zeros((198, 1), device=torch.device('cpu'), dtype=torch.float), requires_grad=True)
        self.Z_w1 = torch.zeros(self.w1.size(), device=self.device, dtype=torch.float)
        self.Z_b1 = torch.zeros(self.b1.size(), device=self.device, dtype=torch.float)
        self.Z_w2 = torch.zeros(self.w2.size(), device=self.device, dtype=torch.float)
        self.Z_b2 = torch.zeros(self.b2.size(), device=self.device, dtype=torch.float)
        
        #Critic part
        self.W=Variable(torch.zeros((1, 198), device=torch.device('cpu'), dtype=torch.float), requires_grad=True)
        self.B=Variable(torch.zeros((1, 1), device=torch.device('cpu'), dtype=torch.float), requires_grad=True)
        self.Z_W = torch.zeros(self.W.size(), device=self.device, dtype=torch.float)
        self.Z_B = torch.zeros(self.B.size(), device=self.device, dtype=torch.float)
        #Actor part
        self.theta=Variable(torch.zeros((1, 198), device=torch.device('cpu'), dtype=torch.float), requires_grad=True)
        #self.thetaB=Variable(torch.zeros((1, 1), device=torch.device('cpu'), dtype=torch.float), requires_grad=True)
        self.Z_theta = torch.zeros(self.theta.size(), device=self.device, dtype=torch.float)
       # self.Z_thetaB = torch.zeros(self.thetaB.size(), device=self.device, dtype=torch.float)
        
#        self.theta=np.zeros(198)
#        self.Z_theta=np.zeros(198)
        
        self.y_sigmoid = 0
        self.target=0
        self.oldtarget=0
        
        self.alpha1 = 0.001
        self.alpha2 = 0.001
        self.c_alpha = 0.001
        self.a_alpha = 0.001
        
        self.gamma=1
        self.delta=0
        
        self.probs=0
        self.grad_ln_pi=0
        
    def createNewFeatures(self, x):
        x_prime = Variable(torch.tensor(x, dtype=torch.float, device=self.device)).view(198, 1)
        
        #First layer
        x1=torch.mm(self.w1, x_prime) + self.b1
        x1_sigmoid=x1.sigmoid()
        #Second layer
        x2=torch.mm(self.w2, x1_sigmoid) + self.b2
        x2_sigmoid=x2.sigmoid()
        
        return x2_sigmoid
    
    def critic_forward(self, xnew, xold):
        #First evaluate the new board xnew
#        x_prime = Variable(torch.tensor(xnew, dtype=torch.float, device=self.device)).view(198, 1)
#        
#        #First layer
#        x1=torch.mm(self.w1, x_prime) + self.b1
#        x1_sigmoid=x_1.sigmoid()
#        #Second layer
#        x2=torch.mm(self.w2, x1_sigmoid) + self.b2
#        x2_sigmoid=x2.sigmoid()
        x=self.createNewFeatures(xnew)
        #Now we have the features, calculate final value using W
        y=torch.mm(self.W, x) + self.B
        self.y_sigmoid=y.sigmoid()
        self.target = self.y_sigmoid.detach().cpu().numpy()
        
        #Now avaluate the old board aswell
#        x_prime = Variable(torch.tensor(xold, dtype=torch.float, device=self.device)).view(198, 1)
#        
#        #First layer
#        x1=torch.mm(self.w1, x_prime) + self.b1
#        x1_sigmoid=x_1.sigmoid()
#        #Second layer
#        x2=torch.mm(self.w2, x1_sigmoid) + self.b2
#        x2_sigmoid=x2.sigmoid()

        x=self.createNewFeatures(xold)
        #Now we have the features, calculate final value using W
        y=torch.mm(self.W,x) + self.B
        self.y_sigmoid=y.sigmoid()
        self.oldtarget = self.y_sigmoid.detach().cpu().numpy()
        
        #return self.y_sigmoid
        
    #X is a list of possible afterstates, Z is a list of possible moves
    def actor_forward(self, X):#, Z):
       # exp=[]
        values=torch.zeros(len(X))
        c=0
        for x in X:
#            x_prime = Variable(torch.tensor(x, dtype=torch.float, device=self.device)).view(198, 1)
#            
#            #First layer
#            x1=torch.mm(self.w1, x_prime) + self.b1
#            x1_sigmoid=x_1.sigmoid()
#            #Second layer
#            x2=torch.mm(self.w2, x1_sigmoid) + self.b2
#            x2_sigmoid=x2.sigmoid()
            
            x_prime=self.createNewFeatures(x)
            
            #Now we have the features, calculate final value using W
            y=torch.mm(self.theta, x_prime) # + self.B
#            y=np.matmul(self.theta, x_prime.detach().numpy())
            #exp.append(np.exp(y))
#            values.append(y)
            values[c]=y
            c+=1
        #sum=np.sum(exp)
       # probs=exp/sum
        #self.probs=self.softmax(values)
        probs=values.softmax(dim=0)
        log_probs=values.log_softmax(dim=0)
#        move=choice(Z, p=probs)
#        return move
        return probs, log_probs
    
    def softmax2(self, values):
        exp=torch.zeros(len(values))
        #exp=[]
        c=0
        for v in values:
            #v=v.detach().numpy()
            #exp.append(torch.exp(v))
            exp[c]=torch.exp(v)
            c+=1
        #probs=[]
        sum=torch.sum(exp).item()
#        for e in exp:
#            probs.append(e/sum)
        probs=exp/sum
        probs=probs/(torch.sum(probs).item())
        return probs
    
    def softmax(self, values):
        exp=[]
        sum=0
        probs=[]
        for v in values:
            exp.append(np.exp(v))
        sum=np.sum(exp)
        probs=exp/sum
        return probs
    
    def critic_back(self, R):
        self.y_sigmoid.backward()
        
        self.Z_w2 = self.gamma * self.lam * self.Z_w2 + self.w2.grad.data
        self.Z_b2 = self.gamma * self.lam * self.Z_b2 + self.b2.grad.data
        self.Z_w1 = self.gamma * self.lam * self.Z_w1 + self.w1.grad.data
        self.Z_b1 = self.gamma * self.lam * self.Z_b1 + self.b1.grad.data
        
        self.Z_B = self.gamma * self.lam * self.Z_B + self.B.grad.data
        
        self.w2.grad.data.zero_()
        self.b2.grad.data.zero_()
        self.w1.grad.data.zero_()
        self.b1.grad.data.zero_()
        self.W.grad.data.zero_()
        self.B.grad.data.zero_()
        
        self.delta=R+self.gamma*self.target-self.oldtarget
        
        self.w1.data = self.w1.data + self.alpha1 * self.delta * self.Z_w1
        self.b1.data = self.b1.data + self.alpha1 * self.delta * self.Z_b1
        self.w2.data = self.w2.data + self.alpha2 * self.delta * self.Z_w2
        self.b2.data = self.b2.data + self.alpha2 *self.delta * self.Z_b2
        
        self.W.data = self.W.data + self.c_alpha * self.delta * self.Z_W
        self.B.data = self.B.data + self.c_alpha * self.delta * self.Z_B
        
        
        
     #xCurr is the currents state, X is a list of afterstates       
    def actor_back(self, xCurr, X, log_probs):
        
        log_probs.backward()
        self.Z_theta=self.gamma*self.lam*self.Z_theta + self.theta.grad.data
        
        self.theta.grad.data.zero_()
        
        self.theta.data=self.theta+self.a_alpha*self.delta * self.Z_theta
#        self.theta.grad.data.zero_()
#        self.Z_theta=self.gamma * self.lam * self.Z_theta + self.ln_grad_pi(xCurr, X)
#        self.theta=self.theta+self.a_alpha*self.delta * self.Z_theta
        
        
    #xCurr is the currents state, X is a list of afterstates    
    def ln_grad_pi(self, xCurr, X):
        xCurr=self.createNewFeatures(xCurr)
        sum=0
        c=0
        for p in self.probs:
            sum+=p * self.createNewFeatures(X[c]).detach().numpy()
        return xCurr.detach().numpy()-sum
        
        
    
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
    
def action(net, board_copy,dice,player,i):
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy
    
    # check out the legal moves available for the throw
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player)
    
    # if there are no moves available
    if len(possible_moves) == 0: 
        return [] 
    
    #Get probabilities of each move
    feature_boards=[]
    for b in possible_boards:
        feature_boards.append(getFeatures(b, player))
    #feature_boards=getFeatures(possible_boards, player)
   # probs=net.actor_forward(possible_boards)
    probs, log_probs=net.actor_forward(feature_boards)
    #probs=probs
#    probs2=[]
#    for p in probs:
#        probs2.append(p.item())
    #Pick a move with given the probabilities probs
    index=np.arange(0, len(possible_boards))
    probs=probs.detach().numpy() #!!!!!!!!!!!!!!!!!!!!!!!!!!
    i=choice(index, p=probs)
    move=possible_moves[i]
    newBoard=possible_boards[i]
    R=0
    if(Backgammon.game_over(newBoard)):
        R=1
    
    #forward critic
    net.critic_forward(getFeatures(newBoard, player), getFeatures(board_copy,player))
    net.critic_back(R)
    net.actor_back(getFeatures(board_copy, player), feature_boards, log_probs[i])
    
    return move
    
    
    
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