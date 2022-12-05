# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.captureAgents import CaptureAgent
from contest.game import Directions
from contest.util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='ourAttackerAgent', second='ourDefensiveAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class GeneralAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.width = game_state.data.layout.width
        self.height = game_state.data.layout.height
        self.frontier = []
        if self.red:
            x = (self.width//2) -1
            for y in range(1,self.height-1):
                self.frontier.append((x,y))
        else: 
            x = (self.width//2)
            for y in range(1,self.height-1):
                self.frontier.append((x,y))
        
        self.currentFood = game_state.get_agent_state(self.index).num_carrying      # Food that hasn't been secured yet
                
    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

    def get_position(self, game_state):
        return game_state.get_agent_state(self.index).get_position()
    
    
    def min_dist_home(self, game_state):     # For saving some eaten dots
        min_dist = 100000
        my_pos = (int(self.get_position(game_state)[0]), int(self.get_position(game_state)[1]))
        for i in self.frontier:
            if not game_state.has_wall(i[0], i[1]):      # Otherwise it raises an error
                dist = self.get_maze_distance(my_pos, i)
                if dist < min_dist:
                    min_dist = dist
                
        return min_dist
    
    def is_home(self, game_state):
        
        if self.red:
            return self.get_position(game_state)[0] <= self.frontier[0][0]
        else:
            return self.get_position(game_state)[0] >= self.frontier[0][0]
            
    def are_there_walls(self, game_state, pos1, pos2):
        count = 0
        if pos1[0] > pos2[0]:
            if pos1[1] > pos2[1]:
                for i in range(int(pos2[0]), int(pos1[0])):
                    for j in range(int(pos2[1]), int(pos1[1])):
                        if game_state.has_wall(i,j):
                            count+= 1
            else : 
                for i in range(int(pos2[0]), int(pos1[0])):
                    for j in range(int(pos1[1]), int(pos2[1])):
                        if game_state.has_wall(i,j):
                            count+= 1
        else:
            if pos1[1] > pos2[1]:
                for i in range(int(pos1[0]), int(pos2[0])):
                    for j in range(int(pos2[1]), int(pos1[1])):
                        if game_state.has_wall(i,j):
                            count+= 1
            else : 
                for i in range(int(pos1[0]), int(pos2[0])):
                    for j in range(int(pos1[1]), int(pos2[1])):
                        if game_state.has_wall(i,j):
                            count+= 1
        return count

    def is_wall_between(self, game_state, pos1, pos2):
        if int(pos1[0]) == int(pos2[0]):
            if pos1[1] > pos2[1]:
                for j in range(int(pos2[1]), int(pos1[1])):
                    if game_state.has_wall(int(pos1[0]),j):
                        return True
            else :
                for i in range(int(pos2[0]), int(pos1[0])):
                    for j in range(int(pos1[1]), int(pos2[1])):
                        if game_state.has_wall(int(pos1[0]),j):
                            return True
        
        elif int(pos1[1]) == int(pos2[1]):
            if pos1[0] > pos2[0]:
                for i in range(int(pos2[0]), int(pos1[0])):
                    if game_state.has_wall(i, int(pos1[1])):
                        return True
            else :
                for i in range(int(pos1[0]), int(pos2[0])):
                    if game_state.has_wall(i, int(pos1[1])):
                        return True
            
        return False
                       





class ourAttackerAgent(GeneralAgent):
         
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)  # self.getScore(successor)
        features['no_way_out'] = 0
        features['attack_ghost'] = 0
        features['go_for_capsule'] = 0
        currentFood = game_state.get_agent_state(self.index).num_carrying
        
        # Compute distance to the nearest food
        my_pos = successor.get_agent_state(self.index).get_position()

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
            
        for i in [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]:
            
            if i.get_position() != None and not i.is_pacman and game_state.get_agent_state(self.index).is_pacman and i.scared_timer < 5: # (for the offensive, the defensive should be aggresive towards pacman)
                current_ghost_pos = i.get_position()
                if self.get_maze_distance(my_pos, current_ghost_pos) < self.get_maze_distance(self.get_position(game_state), current_ghost_pos) or self.are_there_walls(game_state, my_pos, current_ghost_pos) > 3:
                    features['run_from_ghost'] = -100
                elif self.get_maze_distance(my_pos, current_ghost_pos) > self.get_maze_distance(self.get_position(game_state), current_ghost_pos) and not self.is_wall_between(game_state, my_pos, current_ghost_pos):
                    features['run_from_ghost'] = 100
                    
                    if len(successor.get_legal_actions(self.index)) < 3:
                        features['no_way_out'] = -1
            
            elif i.get_position() != None and i.scared_timer > 10:         # Attack ghost!
                current_ghost_pos = i.get_position()
                if self.get_maze_distance(my_pos, current_ghost_pos) < self.get_maze_distance(self.get_position(game_state), current_ghost_pos):
                    features['attack_ghost'] = i.scared_timer
        
            if currentFood > 5 and i.scared_timer < 3:
                if self.min_dist_home(successor) < self.min_dist_home(game_state):
                    features['secure_food'] = -self.min_dist_home(successor)
                else:
                    features['secure_food'] = self.min_dist_home(successor)
            else:
                features['secure_food'] = 0     # Feature = 0 -> deactivated for this situation
            
        for cap in self.get_capsules(game_state):
            if self.get_maze_distance(self.get_position(game_state),cap) < 5:
                if self.get_maze_distance(my_pos,cap) < self.get_maze_distance(self.get_position(game_state),cap):
                    features['go_for_capsule'] = 1
                   
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'no_way_out': 100000000, 'run_from_ghost': 10, 'distance_to_food': -1, 'secure_food': -1000, 'attack_ghost': 100000, 'go_for_capsule': 10000}






class ourDefensiveAgent(GeneralAgent):

  def get_features(self, game_state, action):
      features = util.Counter()
      successor = self.get_successor(game_state, action)
      defendFood = self.get_food_you_are_defending(game_state).as_list()
      my_state = successor.get_agent_state(self.index)
      successor_pos = my_state.get_position()


      # Computes whether we're on defense (1) or offense (0)
      features['on_defense'] = 1
      if my_state.is_pacman: features['on_defense'] = 0

      # Computes distance to invaders we can see
      enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
      invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
      features['num_invaders'] = len(invaders)
      
      if len(invaders) > 0:
          dists = [self.get_maze_distance(successor_pos, a.get_position()) for a in invaders]
          features['invader_distance'] = min(dists)

      if action == Directions.STOP: features['stop'] = 1
      rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
      if action == rev: features['reverse'] = 1

      #If in the next position there is food
      
      for i in [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]:
            
          if i.get_position() != None and i.is_pacman and not successor.get_agent_state(self.index),is_pacman:
            current_pac_pos = i.get_position()
            if self.get_maze_distance(successor_pos, current_pac_pos) < self.get_maze_distance(self.get_position(game_state), current_pac_pos) and not successor.get_agent_state(self.index).scared_timer > 3:
                features['attack_pacman'] = 10000
            elif self.get_maze_distance(successor_pos, current_pac_pos) > self.get_maze_distance(self.get_position(game_state), current_pac_pos):
                features['attack_pacman'] = -1000
            elif successor.get_agent_state(self.index).scared_timer > 3 and self.get_maze_distance(successor_pos, current_pac_pos) > self.get_maze_distance(self.get_position(game_state), current_pac_pos):
                features['attack_pacman'] = 10000
                
      
      return features

  def get_weights(self, game_state, action):
      return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2, 'attack_pacman': 1000}


