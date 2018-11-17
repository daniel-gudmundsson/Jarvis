#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The intelligent agent
see flipped_agent for an example of how to flip the board in order to always
perceive the board as player 1
"""

"""
This agent uses one hot encoding for the board and a shared network for the actor and critic.
discovers new features. Uses correct update hopefully
"""
import numpy as np
import Backgammon
import torch
from torch.autograd import Variable
from numpy.random import choice

class net():
    def __init__(self):
        self.device = torch.device('cuda')
        self.w1 = Variable(torch.randn(99, 832, device=self.device , dtype=torch.float), requires_grad=True)
        self.b1 = Variable(torch.zeros((99, 1), device=self.device , dtype=torch.float), requires_grad=True)
        
        self.w2 = Variable(torch.randn(99, 99, device=self.device , dtype=torch.float), requires_grad=True)
        self.b2 = Variable(torch.zeros((99, 1), device=self.device , dtype=torch.float), requires_grad=True)
        
        self.Z_w1 = torch.zeros(self.w1.size(), device=self.device, dtype=torch.float)
        self.Z_b1 = torch.zeros(self.b1.size(), device=self.device, dtype=torch.float)
        self.Z_w2 = torch.zeros(self.w2.size(), device=self.device, dtype=torch.float)
        self.Z_b2 = torch.zeros(self.b2.size(), device=self.device, dtype=torch.float)
        
        ###The critic
        self.W = Variable(torch.randn(1, 99, device=self.device , dtype=torch.float), requires_grad=True)
        self.B = Variable(torch.zeros((1, 1), device=self.device , dtype=torch.float), requires_grad=True)
        self.Z_W = torch.zeros(self.W.size(), device = self.device, dtype = torch.float)
        self.Z_B = torch.zeros(self.B.size(), device = self.device, dtype = torch.float)
        
        ###The actor
        self.theta = Variable(torch.randn(1, 99, device=self.device , dtype=torch.float), requires_grad=True)
        self.Z_theta = torch.zeros(self.theta.size(), device = self.device, dtype = torch.float)
        
        ###Stuff we need
        self.target=0
        self.oldtarget=0
        self.y_sigmoid=0
        self.hidden=0
        
        ###Parameters
        self.alpha1=0.01
        self.alpha2=0.01
        self.alphaC=0.01
        self.alphaA=0.001
        
        self.lamC=1
        self.lamA=1
        
        self.gamma=1
        
        self.xFlipOld=flip_board(np.copy(Backgammon.init_board()))
        self.xFlipNew=flip_board(np.copy(Backgammon.init_board()))
        self.xold=Backgammon.init_board()
        self.xnew=Backgammon.init_board()
        
        self.xtheta=None
        self.flipxtheta=0
        
        self.firstMove=True
        
    def reset_old_boards(self):
        self.xFlipOld=flip_board(np.copy(Backgammon.init_board()))
        self.xold=Backgammon.init_board()
        
    def zero_el(self):
        self.Z_w1 = torch.zeros(self.w1.size(), device=self.device, dtype=torch.float)
        self.Z_b1 = torch.zeros(self.b1.size(), device=self.device, dtype=torch.float)
        self.Z_w2 = torch.zeros(self.w2.size(), device=self.device, dtype=torch.float)
        self.Z_b2 = torch.zeros(self.b2.size(), device=self.device, dtype=torch.float)
        self.Z_W = torch.zeros(self.W.size(), device = self.device, dtype = torch.float)
        self.Z_B = torch.zeros(self.B.size(), device = self.device, dtype = torch.float) 
        self.Z_theta = torch.zeros(self.theta.size(), device = self.device, dtype = torch.float)
        
        
    def forward(self, x, isActor=False):
        if isActor:
            x = Variable(torch.tensor(x, dtype=torch.float, device=self.device)).view(832, len(x))
        else:
            x = Variable(torch.tensor(x, dtype=torch.float, device=self.device)).view(832, 1)
        x1=torch.mm(self.w1, x) + self.b1
        x1_sigmoid=x1.sigmoid()
        
        x2=torch.mm(self.w2, x1_sigmoid) + self.b2
        x2_sigmoid=x2.sigmoid()
        self.hidden=x2_sigmoid
        
        return x2_sigmoid
        

    def critic(self, x):
        x=self.forward(x)
        y=torch.mm(self.W, x) + self.B
        self.y_sigmoid=y.sigmoid()
        
        return self.y_sigmoid
    
    def actor(self, X, moves):
        X_prime = Variable(torch.tensor(X, dtype=torch.float, device=self.device))
        
        X_prime2=self.forward(X_prime, isActor=True)
        pi = torch.mm(self.theta,X_prime2).softmax(1)
        
        xtheta_mean = torch.sum(torch.mm(X_prime2,torch.diagflat(pi)),1)
        #xtheta_mean = torch.sum(torch.mm(torch.diagflat(pi),X_prime),1)
        xtheta_mean = torch.unsqueeze(xtheta_mean,1)
        self.xtheta=xtheta_mean
        m = torch.multinomial(pi, 1)
        
        return m, xtheta_mean
           
        
    def update(self, player):
        if player==1:
            ###new board afterstate value
            y_sigmoid=self.critic(oneHot(self.xnew))
            target = y_sigmoid.detach().cpu().numpy()
            
            ###For the old board
            y_sigmoid=self.critic(oneHot(self.xold))
            oldtarget = y_sigmoid.detach().cpu().numpy()
            
            delta=0 + self.gamma * target - oldtarget
            delta=torch.tensor(delta, dtype=torch.float, device=self.device)
            ###The critic
            y_sigmoid.backward()
            self.Z_w1 = self.gamma * self.lamC * self.Z_w1 + self.w1.grad.data
            self.Z_b1 = self.gamma * self.lamC * self.Z_b1 + self.b1.grad.data
            self.Z_w2 = self.gamma * self.lamC * self.Z_w2 + self.w2.grad.data
            self.Z_b2 = self.gamma * self.lamC * self.Z_b2 + self.b2.grad.data
            self.Z_W = self.gamma * self.lamC * self.Z_W + self.W.grad.data
            self.Z_B = self.gamma * self.lamC * self.Z_B + self.B.grad.data
        
            self.w1.grad.data.zero_()
            self.b1.grad.data.zero_()
            self.w2.grad.data.zero_()
            self.b2.grad.data.zero_()
            self.W.grad.data.zero_()
            self.B.grad.data.zero_()
        
            self.w1.data = self.w1.data + self.alpha1 * delta * self.Z_w1
            self.b1.data = self.b1.data + self.alpha1 * delta * self.Z_b1
            self.w2.data = self.w2.data + self.alpha2 * delta * self.Z_w2
            self.b2.data = self.b2.data + self.alpha2 * delta * self.Z_b2
            self.W.data = self.W.data + self.alphaC * delta * self.Z_W
            self.B.data = self.B.data + self.alphaC * delta * self.Z_B
            
            ###The actor
            x=self.forward(oneHot(self.xold))
            grad_ln_pi = x - self.xtheta
            self.theta.data = self.theta.data + self.alphaA*delta*grad_ln_pi.view(1,len(grad_ln_pi))
            
        else:
            ###new board afterstate value
            y_sigmoid=self.critic(oneHot(self.xFlipNew))
            target = y_sigmoid.detach().cpu().numpy()
            
            ###For the old board
            y_sigmoid=self.critic(oneHot(self.xFlipOld))
            oldtarget = y_sigmoid.detach().cpu().numpy()
            
            delta=0 + self.gamma * target - oldtarget
            delta=torch.tensor(delta, dtype=torch.float, device=self.device)
            ###The critic
            y_sigmoid.backward()
            self.Z_w1 = self.gamma * self.lamC * self.Z_w1 + self.w1.grad.data
            self.Z_b1 = self.gamma * self.lamC * self.Z_b1 + self.b1.grad.data
            self.Z_w2 = self.gamma * self.lamC * self.Z_w2 + self.w2.grad.data
            self.Z_b2 = self.gamma * self.lamC * self.Z_b2 + self.b2.grad.data
            self.Z_W = self.gamma * self.lamC * self.Z_W + self.W.grad.data
            self.Z_B = self.gamma * self.lamC * self.Z_B + self.B.grad.data
        
            self.w1.grad.data.zero_()
            self.b1.grad.data.zero_()
            self.w2.grad.data.zero_()
            self.b2.grad.data.zero_()
            self.W.grad.data.zero_()
            self.B.grad.data.zero_()
        
            self.w1.data = self.w1.data + self.alpha1 * delta * self.Z_w1
            self.b1.data = self.b1.data + self.alpha1 * delta * self.Z_b1
            self.w2.data = self.w2.data + self.alpha2 * delta * self.Z_w2
            self.b2.data = self.b2.data + self.alpha2 * delta * self.Z_b2
            self.W.data = self.W.data + self.alphaC * delta * self.Z_W
            self.B.data = self.B.data + self.alphaC * delta * self.Z_B
            
            ###The actor
            x=self.forward(oneHot(self.xFlipOld))
            grad_ln_pi = x - self.flipxtheta
            self.theta.data = self.theta.data + self.alphaA*delta*grad_ln_pi.view(1,len(grad_ln_pi))
        
    def gameOverUpdate(self, R):
        target=0
        ###For the old board
        y_sigmoid=self.critic(oneHot(self.xold))
        oldtarget = y_sigmoid.detach().cpu().numpy()
        
        delta=R + self.gamma * target - oldtarget
        delta=torch.tensor(delta, dtype=torch.float, device=self.device)
        ###The critic
        y_sigmoid.backward()
        self.Z_w1 = self.gamma * self.lamC * self.Z_w1 + self.w1.grad.data
        self.Z_b1 = self.gamma * self.lamC * self.Z_b1 + self.b1.grad.data
        self.Z_w2 = self.gamma * self.lamC * self.Z_w2 + self.w2.grad.data
        self.Z_b2 = self.gamma * self.lamC * self.Z_b2 + self.b2.grad.data
        self.Z_W = self.gamma * self.lamC * self.Z_W + self.W.grad.data
        self.Z_B = self.gamma * self.lamC * self.Z_B + self.B.grad.data
    
        self.w1.grad.data.zero_()
        self.b1.grad.data.zero_()
        self.w2.grad.data.zero_()
        self.b2.grad.data.zero_()
        self.W.grad.data.zero_()
        self.B.grad.data.zero_()
    
        self.w1.data = self.w1.data + self.alpha1 * delta * self.Z_w1
        self.b1.data = self.b1.data + self.alpha1 * delta * self.Z_b1
        self.w2.data = self.w2.data + self.alpha2 * delta * self.Z_w2
        self.b2.data = self.b2.data + self.alpha2 * delta * self.Z_b2
        self.W.data = self.W.data + self.alphaC * delta * self.Z_W
        self.B.data = self.B.data + self.alphaC * delta * self.Z_B
        
        ###The actor
        x=self.forward(oneHot(self.xold))
        grad_ln_pi = x - self.xtheta
        self.theta.data = self.theta.data + self.alphaA*delta*grad_ln_pi.view(1,len(grad_ln_pi))

        ########################################################################
        ########################################################################
        
        ###new board afterstate value
       
        target = 0
        
        ###For the old board
        y_sigmoid=self.critic(oneHot(self.xFlipOld))
        oldtarget = y_sigmoid.detach().cpu().numpy()
        
        delta=(1-R) + self.gamma * target - oldtarget
        delta=torch.tensor(delta, dtype=torch.float, device=self.device)
        ###The critic
        y_sigmoid.backward()
        self.Z_w1 = self.gamma * self.lamC * self.Z_w1 + self.w1.grad.data
        self.Z_b1 = self.gamma * self.lamC * self.Z_b1 + self.b1.grad.data
        self.Z_w2 = self.gamma * self.lamC * self.Z_w2 + self.w2.grad.data
        self.Z_b2 = self.gamma * self.lamC * self.Z_b2 + self.b2.grad.data
        self.Z_W = self.gamma * self.lamC * self.Z_W + self.W.grad.data
        self.Z_B = self.gamma * self.lamC * self.Z_B + self.B.grad.data
    
        self.w1.grad.data.zero_()
        self.b1.grad.data.zero_()
        self.w2.grad.data.zero_()
        self.b2.grad.data.zero_()
        self.W.grad.data.zero_()
        self.B.grad.data.zero_()
    
        self.w1.data = self.w1.data + self.alpha1 * delta * self.Z_w1
        self.b1.data = self.b1.data + self.alpha1 * delta * self.Z_b1
        self.w2.data = self.w2.data + self.alpha2 * delta * self.Z_w2
        self.b2.data = self.b2.data + self.alpha2 * delta * self.Z_b2
        self.W.data = self.W.data + self.alphaC * delta * self.Z_W
        self.B.data = self.B.data + self.alphaC * delta * self.Z_B
        
        ###The actor
        x=self.forward(oneHot(self.xFlipOld))
        grad_ln_pi = x - self.flipxtheta
        self.theta.data = self.theta.data + self.alphaA*delta*grad_ln_pi.view(1,len(grad_ln_pi))
            
            
            
            

def action(net, board_copy,dice,player,i, learn=True):
    # the champion to be
    # inputs are the board, the dice and which player is to move
    # outputs the chosen move accordingly to its policy
    
    if player == -1: board_copy = flip_board(board_copy) ##Flip the board
    # check out the legal moves available for the throw
    if(player==1):
        xold=net.xold
        net.xnew=board_copy
    else: ########################################################################
        xold=net.xFlipOld
        net.xFlipNew=board_copy
        
    possible_moves, possible_boards = Backgammon.legal_moves(board_copy, dice, player=1)
    
    # if there are no moves available
    if len(possible_moves) == 0: 
        return [] 
    
    one_hot=[]
    for b in possible_boards:
        one_hot.append(oneHot(b))
    
    if learn:
        if not net.firstMove:
            net.update(player)
    
    m, xtheta =net.actor(one_hot, possible_moves)
    if  player==1:
        net.xtheta=xtheta
    else:
        net.flipxtheta=xtheta
    
    move=possible_moves[m]
    newBoard=possible_boards[m]
  
#    if learn:
#        if not net.firstMove:
#            net.update(player)
        
    if player == -1: move = flip_move(move) ###Flip the move
    
    if player==1:
        net.xold=board_copy
    else:
        net.xFlipOld=board_copy
        net.firstMove=False
    
    return move



###One hot encoding for the board
def oneHot(board):
    one=np.zeros(16*26*2)
    
    ### Player 1
    for i in range(0,25):
        v=board[i]
        if v>0:
            #index=v+i*16
            one[v+(i-1)*16]=1
        else:
            one[(i-1)*16]=1
    
    numJail=board[25]
    one[16*24+numJail]=1
    
    numOffBoard=board[27]
    one[16*25+numOffBoard]=1
    
    
    ### Player -1
    for i in range (1,25):
        v=board[i]
        if v<0:
            one[v+16*26+(i-1)*16]=1
        else:
            one[16*26+(i-1)*16]=1
    
    numJail=board[26]
    one[16*24*2+numJail]=1
    
    numOffBoard=board[28]
    one[16*25*2+numOffBoard]=1
    
    return one

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

def flip_board(board_copy):
    #flips the game board and returns a new copy
    idx = np.array([0,24,23,22,21,20,19,18,17,16,15,14,13,
    12,11,10,9,8,7,6,5,4,3,2,1,26,25,28,27])
    flipped_board = -np.copy(board_copy[idx])
        
    return flipped_board

def flip_move(move):
    if len(move)!=0:
        for m in move:
            for m_i in range(2):
                m[m_i] = np.array([0,24,23,22,21,20,19,18,17,16,15,14,13,
                                12,11,10,9,8,7,6,5,4,3,2,1,26,25,28,27])[m[m_i]]        
    return move
