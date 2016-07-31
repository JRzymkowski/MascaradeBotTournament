# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 13:50:41 2016

Author: Michał Kaftanowicz, http://kaftanowicz.com
Inspiration from: Przemysław Kowalczyk, http://pkowalczyk.pl/

Disclaimer:
Mascarade board game was created by Bruno Faidutti (http://faidutti.com/blog/)
Rules encoded and quoted in this program are referenced
for educational purposes only.

 Masacarde card game bot and bot tournament
 Assumptions:
 - cards from basic edition are used
 - number of players is between 4 and 13
 - there is a public game history log
   (game as seen by a non-participating observator)
   and the true game history log (for aftermath analysis)
 - the tournament environment stores the game state
   and after each bot's action adds it to the true and public
   (after masking some values) game history logs
 - each bot stores its internal game representation in
   a belief data structure and a private game history log
   (describing game as seen from his perspective)
 - a bot is a function from the domain
   {public game history, private game history, beliefs, action mode}
   to codomain {beliefs, action}
 - the tournament environment updates the true and the public
   game history logs on a bot's action
 - the tournament environment calls each bot subsequently, specifying
   the action mode according to the game rules
   
"""

import numpy as np
import random

def PlayerToMyLeft(myPlayerNumber, numberOfPlayers):
    return((myPlayerNumber+1) % numberOfPlayers)
    
def PlayerToMyRight(myPlayerNumber, numberOfPlayers):
    return((myPlayerNumber-1) % numberOfPlayers)
    
def SinkhornKnoppBalance(squareMx, excludedRow, excludedColumn):
    """
     Takes a matrix (array), excludes some row and some column
     and balances the remaining matrix so that the sum of
     every row and the sum of every column is equal to 1,
     by alternatingly normalizing its rows and columns.
     Stops when the rows' total squared error is less than epsilon.
     (Columns get normalized properly for sure)
    """
    arrayDim = squareMx.shape[0]
    epsilon = 0.01
    error = epsilon + 1 # so that the while loop starts
    #iteration = 0
    while error > epsilon:
        #iteration = iteration + 1
        error = 0
        for i in range(arrayDim):
            if i != excludedRow:
                squareMx[i,:] = squareMx[i,:] / np.sum(squareMx[i,:])
        for j in range(arrayDim):
            if j != excludedColumn:
                squareMx[:,j] = squareMx[:,j] / np.sum(squareMx[:,j])
                error = error + (np.sum(squareMx[j,:]) -1) ** 2
        #print('Iteration ', iteration, ', total sqrd error = ', error)
    return(squareMx)

def UpdateBeliefsOnCharacterReveal(beliefs, numberOfRevealedPlayer, 
                                   revealedCharacter, startingAssignmentOfCharacters):   
    numberOfRevealedCharacter = int(np.where(startingAssignmentOfCharacters == revealedCharacter)[0])
    beliefsPosterior = 1*beliefs    
    beliefsPosterior[numberOfRevealedPlayer,:] = 0
    beliefsPosterior[:,numberOfRevealedCharacter] = 0
    beliefsPosterior[numberOfRevealedPlayer,numberOfRevealedCharacter] = 1
    beliefsPosterior = SinkhornKnoppBalance(beliefsPosterior, numberOfRevealedPlayer,
                                   numberOfRevealedCharacter)
    return(beliefsPosterior)

def SwapCards(charactersInPlay, numberOfSwapperI, numberOfSwapperII):
    temp = charactersInPlay[numberOfSwapperI]
    charactersInPlay[numberOfSwapperI] = charactersInPlay[numberOfSwapperII]
    charactersInPlay[numberOfSwapperII] = temp
    return(charactersInPlay)

# SwapCards(['a', 'b', 'c', 'd'], 0, 1)

def UpdateBeliefsOnCardSwap(beliefs, numberOfSwapperI, numberOfSwapperII, 
                            swapSubjectiveProbability):
    beliefsPrior = 1*beliefs
    beliefsPosterior = 1*beliefs  
    beliefsPosterior[numberOfSwapperI,:] = beliefsPrior[numberOfSwapperI,:] * (1-swapSubjectiveProbability) + beliefsPrior[numberOfSwapperII,:] * swapSubjectiveProbability  
    beliefsPosterior[numberOfSwapperII,:] = beliefsPrior[numberOfSwapperII,:] * (1-swapSubjectiveProbability) + beliefsPrior[numberOfSwapperI,:] * swapSubjectiveProbability              
    return(beliefsPosterior)

# actionModes = ('Regular', 'Swap only', 'Challenge the announcer')
# actionTypes = ('Swap my card', 'Look at my card', 'Announce my character')

players = np.array(('BotA', 'BotB', 'BotB', 'BotB'))

def MascaradeTournament(players, numberOfGames):
  # Starting assumptions
  goldCoinsTotal = 194
  characters = np.array(['Judge', 'Bishop', 'King', 'Fool', 'Queen',  
                      'Thief', 'Witch', 'Spy', 'Peasant', 'Peasant', 
                       'Cheat', 'Inquisitor', 'Widow'])
  # Array of characters allowed depending on the number of players,
  # valid for number of players in range 4-13
  charactersAllowedArray = np.array([
    [1,1,1,1,1,1,1,1,1,1], #Judge
    [1,1,1,1,1,1,1,1,1,1], #Bishop
    [0,0,0,0,1,1,1,1,1,1], #King
    [1,1,1,1,1,1,1,1,1,1], #Fool
    [1,1,1,1,1,1,1,1,1,1], #Queen
    [1,0,0,1,0,0,0,0,0,1], #Thief
    [0,1,1,1,1,1,1,1,1,1], #Witch
    [0,0,0,1,0,0,1,1,1,1], #Spy
    [0,0,0,0,1,1,1,1,1,1], #Peasant
    [0,0,0,0,1,1,1,1,1,1], #Peasant
    [1,1,1,0,0,1,1,1,1,1], #Cheat
    [0,0,0,0,0,0,0,1,1,1], #Inquisitor
    [0,0,0,0,0,0,0,0,1,1], #Widow
    ])
  # Initialization of the game
  numberOfPlayers = len(players)
  ## Randomizing the order of players:
  players = np.array(random.sample(list(players), numberOfPlayers))
  activePlayers = np.copy(players)
  numberOfActivePlayers = len(activePlayers)
  ## If there are 4 or 5 players, 2 or 1 cards are placed on the table;
  ## in this case, the table plays a role of a dummy player
  ## who takes no actions and has no coins
  if (numberOfPlayers < 6):
      players = np.append(players, ['Table'] * (6-numberOfPlayers))
      numberOfPlayers = 6
  publicKnowledge = {}
  publicKnowledge['numberOfActivePlayers'] = numberOfActivePlayers
  ## Every player starts with 6 coins
  publicKnowledge['coinsOfPlayers'] = np.array([6] * numberOfActivePlayers)
  publicKnowledge['coinsInBank'] = goldCoinsTotal - sum(publicKnowledge['coinsOfPlayers'])
  publicKnowledge['coinsInCourthouse'] = 0
  ## Choosing allowed character cards depending on the number of players
  charactersAllowedIndex = charactersAllowedArray[:,(numberOfActivePlayers-4)]
  charactersInPlay = characters[np.where(charactersAllowedIndex)]
  ## Random assignment of character cards to players;
  ## later in the game this array will contain
  ## the true assignment of character cards at any moment
  charactersInPlay = np.array(random.sample(list(charactersInPlay), 
                                            numberOfPlayers))
  publicKnowledge['startingAssignmentOfCharacters'] = np.copy(charactersInPlay)
  # At the beginning, character cards assignment is known to all players
  publicKnowledge['revealedCharacters'] = np.array([True] * numberOfPlayers)
  # The following public knowledge items are list updated on each event
  publicKnowledge['announcedCharacters'] = []
  publicKnowledge['challengesToAnnouncers'] = []
  publicKnowledge['declaredCardSwaps'] = []
  # A prior beliefs array is an identity matrix
  # - we have full knowledge about who got which card
  beliefsPrior = np.identity(numberOfPlayers)
  # Every player gets their own private beliefs array
  playersBeliefs = [[beliefsPrior] for k in range(numberOfActivePlayers)]
  privateKnowledge = [{'performedCardSwap':{'eventNumber':-1,
                                            'numberOfSwapperI':None,
                                            'numberOfSwapperII':None},
                       'lookAtMyCard':{'eventNumber':-1,
                                       'myCharacter':None}} \
                       for k in range(numberOfActivePlayers)]
  numberOfTurnsForSwapOnly = 4 # according to the rules
  # Counters' initialization
  turnNumber = 0
  eventNumber = 0
  # Here an actual game starts
  while all(publicKnowledge['coinsOfPlayers'] > 0) and \
        all(publicKnowledge['coinsOfPlayers'] < 13):
      
      for currentPlayer in range(numberOfActivePlayers):
          
          turnNumber = turnNumber + 1
          eventNumber = eventNumber + 1
          actionMode = 'Regular' # default
          
          # Restrictions on types of actions allowed:
          if turnNumber <= numberOfTurnsForSwapOnly:
              actionMode = 'Swap only'
          if publicKnowledge['revealedCharacters'][currentPlayer]:
              actionMode = 'Swap only'
              
          # Current player makes a move
          playerMove = globals()[players[currentPlayer]](
          currentPlayer, publicKnowledge, privateKnowledge[currentPlayer],
          playersBeliefs[currentPlayer], actionMode, eventNumber)
          
          # If player decides to swap their card – or not (under the table):
          if playerMove['actionType'] == 'Swap my card':
              swapTarget = playerMove['actionArgument']
              publicKnowledge['revealedCharacters'][currentPlayer] = False
              publicKnowledge['revealedCharacters'][swapTarget] = False
              if playerMove['actionTrue'] == True:
                  charactersInPlay = SwapCards(charactersInPlay,
                                               currentPlayer, swapTarget)
                  privateKnowledge[currentPlayer]['performedCardSwap'] = \
                  {'eventNumber':eventNumber,
                  'numberOfSwapperI':currentPlayer,
                  'numberOfSwapperII':swapTarget}
          # If player decides to secretly look at their card:
          if playerMove['actionType'] == 'Look at my card':
              privateKnowledge[currentPlayer]['lookAtMyCard'] = \
                  {'eventNumber':eventNumber,
                  'myCharacter':charactersInPlay[currentPlayer]}
              playersBeliefs[currentPlayer] = UpdateBeliefsOnCharacterReveal(
                                              playersBeliefs[currentPlayer],
                                              currentPlayer, 
                                              charactersInPlay[currentPlayer], 
                                              startingAssignmentOfCharacters)
                              
          # If player decides to announce their character:
          if playerMove['actionType'] == 'Announce my character':
              announcedCharacter = playerMove['actionArgument']
  #
  return(trueGameHistory)
  



# Testing beliefs update function
beliefs1

beliefs1 = np.identity(5)
beliefs1 = UpdateBeliefsOnCardSwap(beliefs1, 0, 1, 0.4)
beliefs1 = UpdateBeliefsOnCardSwap(beliefs1, 0, 2, 0.4)
beliefs1 = UpdateBeliefsOnCardSwap(beliefs1, 0, 3, 0.4)
beliefs1 = UpdateBeliefsOnCardSwap(beliefs1, 0, 4, 0.6)
beliefs1 = UpdateBeliefsOnCardSwap(beliefs1, 1, 2, 0.4)
beliefs1 = UpdateBeliefsOnCardSwap(beliefs1, 1, 3, 0.4)
beliefs1 = UpdateBeliefsOnCardSwap(beliefs1, 1, 4, 0.5)
beliefs1 = UpdateBeliefsOnCardSwap(beliefs1, 0, 1, 0.4)
beliefs1 = UpdateBeliefsOnCardSwap(beliefs1, 1, 2, 0.4)
beliefs1 = UpdateBeliefsOnCardSwap(beliefs1, 4, 3, 0.1)
beliefs1 = UpdateBeliefsOnCardSwap(beliefs1, 2, 4, 0.4)
beliefs1
b2 = 1*beliefs1

b2 = UpdateBeliefsOnCharacterReveal(beliefs1, 4, 'A', np.array(['A', 'B', 'C', 'D', 'E']))
b2[2,:] -1

for i in range(5):
    print(np.sum(b2[:,i]))
    print(np.sum(b2[i,:]))
    


    

def BotDraft(myPlayerNumber, publicKnowledge, privateKnowledge, myBeliefs,
         actionMode, eventNumber):
    """
    actionModes = ('Regular', 'Swap only', 'Challenge the announcer')
    actionTypes = ('Swap my card', 'Look at my card', 'Announce my character')
    actionArgument = the number of player with whom to swap cards
    or the name of character to announce
    actionTrue = True or False (whether to actually swap the cards
    or challenge the announcer)
    """
    # Updating beliefs
    if privateKnowledge['lookAtMyCard']['eventNumber'] == (eventNumber - 1):
        myRevealedCharacter = privateKnowledge['lookAtMyCard']['myCharacter']
        myBeliefs = UpdateBeliefsOnCharacterReveal(myBeliefs, 
                    myPlayerNumber, myRevealedCharacter, 
                    publicKnowledge['startingAssignmentOfCharacters'])
    if privateKnowledge['performedCardSwap']['eventNumber'] == (eventNumber - 1):
        numberOfSwapperI = privateKnowledge['performedCardSwap']['numberOfSwapperI']
        numberOfSwapperII = privateKnowledge['performedCardSwap']['numberOfSwapperII']
        myBeliefs = UpdateBeliefsOnCardSwap(myBeliefs, 
                    numberOfSwapperI, numberOfSwapperII, 1):
    # Action
    if actionMode == 'Challenge the announcer':
       # in this case, only the value of actionTrue is used
       actionTrue = False
       return {'privateKnowledge':privateKnowledge, 'myBeliefs':myBeliefs,
               'actionTrue':actionTrue}
    if actionMode == 'Swap only':
       # in this case, only the values of actionArgument and actionTrue
       # are used; actionArgument should be the number of player 
       # with whom to swap cards;
       # if not a number from available range (0 - (number of players-1)),
       # will be ignored and environment will perform (or not) a swap
       # with randomly chosen opponent
       actionArgument = PlayerToMyRight(myPlayerNumber, 
                                        publicKnowledge['numberOfActivePlayers'])
       actionTrue = False
       return {'privateKnowledge':privateKnowledge,'myBeliefs':myBeliefs,
               'actionArgument':actionArgument, 'actionTrue':actionTrue}
    if actionMode == 'Regular':
       
       # If player decides to swap their card – or not (under the table):
       actionType = 'Swap my card'
       actionArgument = PlayerToMyLeft(myPlayerNumber, 
                                        publicKnowledge['numberOfActivePlayers'])
       actionTrue = False
       return {'privateKnowledge':privateKnowledge, 'myBeliefs':myBeliefs,
               'actionType':actionType, actionArgument':actionArgument, 
               'actionTrue':actionTrue}
               
       # If player decides to secretly look at their card:
       actionType = 'Look at my card'
       ## actionArgument and actionTrue are not used in this case
       return {'privateKnowledge':privateKnowledge, 'myBeliefs':myBeliefs,
               'actionType':actionType}
  
       # If player decides to announce their character:
       actionType = 'Announce my character'
       actionArgument = 'Judge'
       ## if actionArgument is not one of the characters in play,
       ## the tournament environment will choose one randomly
       ## actionTrue is not used in this case
       return {'privateKnowledge':privateKnowledge, 'myBeliefs':myBeliefs, 
               'actionType':actionType, 'actionArgument':actionArgument}
    return {None}
   
def UpdatepublicKnowledge(publicKnowledge, action):
    #
    return(publicKnowledge)

def UpdateTrueGameHistory(trueGameHistory, action):
    #
    return(trueGameHistory)
    


'''
o I have a list of tuples such as this:

[(1,"juca"),(22,"james"),(53,"xuxa"),(44,"delicia")]

I want this list for a tuple whose number value is equal to something.

So that if I do search(53) it will return the index value of 2

[i for i, v in enumerate(L) if v[0] == 53]
'''
