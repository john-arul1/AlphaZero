# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 10:45:09 2023

@author: John

# typed from freecodecamp video

tic-tac-toe with MCTS algorithm and NN

The  code is not yet tested

"""


import numpy as np
import math

import torch

print(torch.__version__)


import torch.nn as nn
import torch.nn.functional as F


class TicTacToe:
     
    def __init__(self):
        self.row_count=3
        self.col_count=3
        self.action_size=self.row_count*self.col_count
        
        
        
    def get_initial_state(self):
        return np.zeros((self.row_count,self.col_count)) 
    
    
    
    def get_next_state(self,state,action,player):
        
        row=action  // self.col_count
        column=action % self.row_count
        state [row,column]=player
        return state
    
    
    
    def get_valid_moves(self,state):
        
        return (state.reshape(-1)==0).astype(np.uint8)
        
    
    
    def check_win(self,state,action):
        
        if action == None:
            return False
        
        row=action // self.col_count
        column=action % self.row_count
        player= state[row,column]
        
        return(
            np.sum(state[row,:]) == player*self.col_count
            or np.sum(state[:,column]) == player*self.row_count
            or np.sum(np.diag(state)) ==  player*self.row_count
            or np.sum(np.diag(np.flip(state,axis=0))) ==  player*self.row_count
            )
            
    
    
    def get_value_and_terminated(self,state,action):
        
        if self.check_win(state,action):
        
            return 1, True
        
        if np.sum(self.get_valid_moves(state))==0:
            
            return 0, True
        
        return 0, False  
    
    def get_opponent(self,player):   
       return -player 
   
   
    def get_opponent_value(self,value):
       return -value 
   
   
    
    def change_perspective(self,state,player):
        return state*player
        
        
              
        
      
class Node:
     def __init__(self,game,args,state,parent=None,action_taken=None):
        
         self.game =game
         self.args=args
         self.state=state
         self.parent=parent
         self.action_taken=action_taken
         
         self.children=[]
         self.expandable_moves= game.get_valid_moves(state)
         self.visit_count=0
         self.value_sum =0
         
         
         
     def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0


     def select(self):
         best_child=None
         best_ucb = - np.inf
         for child in self.children:
             ucb= self.get_ucb(child)
             if ucb > best_ucb:
                best_ucb=ucb
                best_child = child
                
         return best_child
     
        
     def get_ucb(self,child):
         
         #0q_value = (1-child.value_sum /(child.visit_count+1))/2
         
         q_value = 1-(child.value_sum /child.visit_count+1)/2

         return q_value+self.args['C']*math.sqrt(math.log(self.visit_count)/child.visit_count)
     
     
     def expand(self):
         action = np.random.choice(np.where(self.expandable_moves==1)[0])
         self.expandable_moves[action]=0 # expanded
         
         child_state=self.state.copy()
         child_state = self.game.get_next_state(child_state,action,1)
         
         child_state = self.game.change_perspective(child_state,player=-1)
         
         child = Node(self.game,self.args,child_state,self,action)
         
         self.children.append(child)
         
         return child
     
        
     def simulate(self):
         
         value,is_terminal = self.game.get_value_and_terminated(self.state,self.action_taken)
         value = self.game.get_opponent_value(value)
         
         
         if is_terminal:
             return value
         
         rollout_state = self.state.copy()
         rollout_player =1
         while True:
             
             valid_moves = self.game.get_valid_moves(rollout_state)
             action= np.random.choice(np.where(valid_moves==1)[0])
             rollout_state = self.game.get_next_state(rollout_state,action,rollout_player)
             value,is_terminal= self.game.get_value_and_terminated(rollout_state,action)
             
             if is_terminal:
                 if rollout_player == -1:
                     value=self.game.get_opponent_value(value)
                 return value
             
             rollout_player= self.game.get_opponent(rollout_player)
             
             
             
     def backpropagate(self,value):
          
          self.value_sum+=value
          self.visit_count+=1
          
          value =self.game.get_opponent_value(value)
           
          
          if self.parent is not None:
              self.parent.backpropagate(value)
              
     
         
  
class MCTS:
    def __init__(self,game,args):
        self.game =game
        self.args=args
     
          
    def search(self,state):
        root=Node(self.game,self.args,state)
        
        for search in range(self.args['num_searches']):
            
            node = root
            while node.is_fully_expanded():
                  node=node.select()
                               
            value,is_terminal =  self.game.get_value_and_terminated(node.state,node.action_taken)
            value=self.game.get_opponent_value(value)
            
            
            #expand and simulate 
            
            if not is_terminal:
                node=node.expand()
                value=node.simulate()
            
            node.backpropagate(value)
           
           
        action_probs=np.zeros(self.game.action_size)
        
        for child in root.children:
            action_probs[child.action_taken]= child.visit_count
        action_probs /= np.sum(action_probs) 
        return action_probs
    
   

        
class ResNet(nn.Module):
        def __init__(self, game, num_resBlocks, num_hidden):
            super().__init__()
            self.startBlock = nn.Sequential( 
                nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_hidden),
                nn.ReLU())
            
            self.backBone = nn.ModuleList(
                [ResBlock(num_hidden) for i in range(num_resBlocks)],
            )
            
            self.policyHead = nn.Sequential(
                nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(32 * game.row_count * game.column_count, game.action_size)
            )
            
            self.valueHead = nn.Sequential(
                nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
                nn.BatchNorm2d(3),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3 * game.row_count * game.column_count, 1),
                nn.Tanh()
            )
            
        def forward(self, x):
            x = self.startBlock(x)
            for resBlock in self.backBone:
                x = resBlock(x)
            policy = self.policyHead(x)
            value = self.valueHead(x)
            return policy, value
        
       
class ResBlock(nn.Module):
      def __init__(self, num_hidden):
            super().__init__()
            self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(num_hidden)
            self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(num_hidden)
            
      def forward(self, x):
            residual = x
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            x += residual
            x = F.relu(x)
            return x
   

              
tictactoe= TicTacToe()
player = 1


args={'C':2.5,'num_searches':2000}

mcts=MCTS(tictactoe,args)



state=tictactoe.get_initial_state()

while True:

  print(state)

  if player == 1:         
        valid_moves = tictactoe.get_valid_moves(state)
        print()
        action=int(input(f"{player}:"))
        
        if valid_moves[action]==0:
            print("invalid action")
            continue
  else:
      neutral_state=tictactoe.change_perspective(state, player)
      mcts_probs=mcts.search(neutral_state)
      action=np.argmax(mcts_probs)
  
  if player ==-1:    
      print(mcts_probs,action)    
  
  state = tictactoe.get_next_state(state, action, player)
  
  value, is_terminal = tictactoe.get_value_and_terminated(state,action)
  
  if is_terminal:
      print(state)
      
      if value==1:
          print("won")
      else:
          print("Draw")
          
      break
  
  
  player = tictactoe.get_opponent(player)
  
                  
                  
                
                    
    
    
    



        
        
