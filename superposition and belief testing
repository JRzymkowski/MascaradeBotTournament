# introduces class superpositionOpt, which represents beliefs about card placement in form of superposition of possible card positions
# class superposition is old version of superpositionOpt, I left it as it's code is clearer. 
# however it's not optimized and might pose problem for large players numbers
# use of superpositionOpt is strongly suggested

# on initialization, superpositionOpt takes one argument - players number
# methods of superpositionOpt:
#   setPlacement(permutation) - permutation is list representing cards of players; sets cards placement
#   reset(); resets the superposition, same as setPlacement([0,1,2,3,...])
#   swap(cards, probability) - cards is a tuple of player numbers, whose cards are being swapped, 
#   probability is subjetive probability of swap; modifies superposition to account for swap
#   reveal(player, card) - player and card are integers; modifies superposition to account for reveal
#   belief(); returns belief matrix generated from superposition

#   additional method: probabilisticReveal(player, card, probability); modifies superposition, as if reveal has happened with  stated probability
#   use with caution. idea was to use that to account for card announcements, but it's not exactly that
# superpositionOpt instances can be added and multiplied by scalars

# examples of class use are below class definitions
# below examples is a working enviornament for prediction testing
# you can test your predictor, just add it to predictors list and add its name to Names
# your predictor has to have methods: reset(), swap(cards), reveal(player, card) and method belief() that returns belief matrix
# bear in mind that predictor method swap doesn't take subjective probability as an argument as it will not be supplemented by testing enviornament

import itertools as it
import numpy as np

playersNumber = 7
class superposition:
    def __init__(self, playersNumber):
        self.playersNumber = playersNumber
        self.possiblePermutations = list(it.permutations(range(playersNumber))) #[0,2,3,1] means player 0 has card 0, player 1 has card 2, etc.
        
        #dictionary with tuples as keys
        #because lists can't be keys in Python, because why would Python lists behave in any way reasonable ever
        #and yes, if you want to modify a tuple, you need to convert it into list, modify the list and then convert it back into tuple
        #and bear in mind that almost everything can be dictionary key in Python
        #heck, I could put numpy as a dictionary key and Python would be okey with that
        #but not lists, nope
        self.cardPlacementSuperposition = {}
        for p in possiblePermutations:
            self.cardPlacementSuperposition[tuple(p)] = 0
            
        self.cardPlacementSuperposition[tuple(range(playersNumber))] = 1 #initialy card placement is [0,1,2,3...]
    
    def setPlacement(self, placement):
        self.cardPlacementSuperposition = {}
        self.cardPlacementSuperposition[tuple(placement)] 
    
    def permutationToMatrix(self, permutation):
        #columns are players, rows are cards
        #matrix represents card placement, not probability
        matrix = np.zeros((self.playersNumber, self.playersNumber))
        for i, p in enumerate(permutation):
            matrix[i][p] = 1
        return matrix
    
    def belief(self):
        #columns are players, rows are cards
        #matrix represents probability of owning card
        belief = np.zeros((self.playersNumber, self.playersNumber))
        for permutation, probability in self.cardPlacementSuperposition.items():
            belief += probability*permutationToMatrix(permutation)
        return belief
        
    def swap(self, cards, subjectiveProbability):
        newSuperposition = {}
        for p in self.possiblePermutations:
            newSuperposition[tuple(p)] = 0
        for permutation, probability in self.cardPlacementSuperposition.items():
            if(probability != 0):
                tempPermutation = list(permutation)
                tempValue = tempPermutation[cards[0]]
                tempPermutation[cards[0]] = tempPermutation[cards[1]]
                tempPermutation[cards[1]] = tempValue
                newPermutation = tuple(tempPermutation)
                
                newSuperposition[newPermutation] += probability*subjectiveProbability
                newSuperposition[permutation] += probability*(1-subjectiveProbability)
        self.cardPlacementSuperposition = dict(newSuperposition)
    
    def reveal(self, player, card):
        newSuperposition = {}
        for p in self.possiblePermutations:
            newSuperposition[tuple(p)] = 0
        probabilitySum = 0
        for permutation, probability in self.cardPlacementSuperposition.items():
            if(permutation[player] == card):
                newSuperposition[permutation] = probability
                probabilitySum += probability
        if probabilitySum == 0:
            print("Error, impossible situation")
            self.cardPlacementSuperposition = {}
        else:
            for p in self.possiblePermutations:
                newSuperposition[tuple(p)] /= probabilitySum
            self.cardPlacementSuperposition = dict(newSuperposition)

### end of class definition

class superpositionOpt:
    #class superposition optimized for sparse probablity vector, i.e. most of permutation has probablity 0
    def __init__(self, playersNumber):
        self.playersNumber = playersNumber
        
        #dictionary with tuples as keys
        #because lists can't be keys in Python, because why would Python lists behave in any way reasonable ever
        #and yes, if you want to modify a tuple, you need to convert it into list, modify the list and then convert it back into tuple
        self.cardPlacementSuperposition = {}          
        self.cardPlacementSuperposition[tuple(range(playersNumber))] = 1 #initialy card placement is [0,1,2,3...]
    
    def setPlacement(self, placement):
        self.cardPlacementSuperposition = {}
        self.cardPlacementSuperposition[tuple(placement)] = 1
        
    def reset(self):
        self.setPlacement(range(self.playersNumber))
    
    def permutationToMatrix(self, permutation):
        #columns are players, rows are cards
        #matrix represents card placement, not probability
        matrix = np.zeros((self.playersNumber, self.playersNumber))
        for i, p in enumerate(permutation):
            matrix[i][p] = 1
        return matrix
    
    def belief(self):
        #columns are players, rows are cards
        #matrix represents probability of owning card
        belief = np.zeros((self.playersNumber, self.playersNumber))
        for permutation, probability in self.cardPlacementSuperposition.items():
            belief += probability*self.permutationToMatrix(permutation)
        return belief
        
    def swap(self, cards, subjectiveProbability):
        newSuperposition = {}
        for permutation, probability in self.cardPlacementSuperposition.items():
            if(probability != 0):
                tempPermutation = list(permutation)
                tempValue = tempPermutation[cards[0]]
                tempPermutation[cards[0]] = tempPermutation[cards[1]]
                tempPermutation[cards[1]] = tempValue
                newPermutation = tuple(tempPermutation)
                
                if(newPermutation in newSuperposition):
                    newSuperposition[newPermutation] += probability*subjectiveProbability
                else:
                    newSuperposition[newPermutation] = probability*subjectiveProbability
                if(permutation in newSuperposition):
                    newSuperposition[permutation] += probability*(1-subjectiveProbability)
                else:
                    newSuperposition[permutation] = probability*(1-subjectiveProbability)
        self.cardPlacementSuperposition = dict(newSuperposition)
    
    def reveal(self, player, card):
        newSuperposition = {}
        probabilitySum = 0
        for permutation, probability in self.cardPlacementSuperposition.items():
            if(permutation[player] == card):
                newSuperposition[permutation] = probability
                probabilitySum += probability
        if probabilitySum == 0:
            print("Error, impossible situation")
            print(player, card)
            print(self.belief())
            self.cardPlacementSuperposition = {}
            return "error"
        else:
            for p in newSuperposition:
                newSuperposition[p] /= probabilitySum
            self.cardPlacementSuperposition = dict(newSuperposition)
            
    def __rmul__(self, coefficient):
        newSuperposition = {}
        for permutation, probability in self.cardPlacementSuperposition.items():
            newSuperposition[permutation] = probability*coefficient
        selfCopy = superpositionOpt(self.playersNumber)
        selfCopy.cardPlacementSuperposition = newSuperposition
        return selfCopy
    
    def __add__(self, element):
        newSuperposition = {}
        allPermutation = set().union(*[self.cardPlacementSuperposition,element.cardPlacementSuperposition])
        for p in allPermutation:
            probability = 0
            if(p in self.cardPlacementSuperposition):
                probability += self.cardPlacementSuperposition[p]
            if(p in element.cardPlacementSuperposition):
                probability += element.cardPlacementSuperposition[p]
            newSuperposition[p] = probability
        selfCopy = superpositionOpt(self.playersNumber)
        selfCopy.cardPlacementSuperposition = newSuperposition
        return selfCopy
    
    def probabilisticReveal(self, player, card, subjectiveProbability):
        afterReveal = superpositionOpt(self.playersNumber)
        afterReveal.cardPlacementSuperposition = self.cardPlacementSuperposition
        afterReveal.reveal(player, card)
        self.cardPlacementSuperposition = (subjectiveProbability*afterReveal+(1-subjectiveProbability)*self).cardPlacementSuperposition
        
    def probabilisticRevealC(self, player, card, subjectiveProbability):
        afterReveal = superpositionOpt(self.playersNumber)
        afterReveal.cardPlacementSuperposition = self.cardPlacementSuperposition
        afterReveal.reveal(player, card)
        self.cardPlacementSuperposition = ( \
        sqrt(2)/(1+subjectiveProbability)*subjectiveProbability*afterReveal+ \
        (1-(subjectiveProbability*sqrt(2)/(1+subjectiveProbability)))*self).cardPlacementSuperposition
        
    def revealNot(self, player, card):
        newSuperposition = {}
        probabilitySum = 0
        for permutation, probability in self.cardPlacementSuperposition.items():
            if(permutation[player] != card):
                newSuperposition[permutation] = probability
                probabilitySum += probability
        if probabilitySum == 0:
            print("Error, impossible situation")
            print(player, card)
            print(self.belief())
            self.cardPlacementSuperposition = {}
            return "error"
        else:
            for p in newSuperposition:
                newSuperposition[p] /= probabilitySum
            self.cardPlacementSuperposition = dict(newSuperposition)
            
    def probabilisticRevealB(self, player, card, subjectiveProbability):
        afterReveal = superpositionOpt(self.playersNumber)
        afterReveal.cardPlacementSuperposition = self.cardPlacementSuperposition
        afterReveal.reveal(player, card)
        
        priorProbability = self.belief()[player, card]
        effectiveProbability = subjectiveProbability*0.5/(1-priorProbability)
        
        self.cardPlacementSuperposition = (effectiveProbability*afterReveal+(1-effectiveProbability)*self).cardPlacementSuperposition
    

### end of class superpositionOpt definition

### here are examples of use of superpositionOpt class

cards = superpositionOpt(playersNumber)

cards.swap([0,1], 0.5)
cards.swap([1,2], 0.7)
cards.swap([4,5], 0.6)
print(cards.belief())

cards.reveal(0, 2)
print(cards.belief())

cards.reset()
cards.swap([0,5], 0.7)
cards.swap([3,2], 0.3)
cards.swap([0,6], 0.5)
cards.swap([1,2], 0.8)
cards.reveal(0, 0)
cards.swap([5,3], 0.6)
print(cards.belief())

### end of examples

### simple/trivial predictors for comparison

class baseline:
    def __init__(self, playersNumber):
        self.playersNumber = playersNumber
        self.matrix = np.zeros((self.playersNumber, self.playersNumber))
        for i in range(self.playersNumber):
            for j in range(self.playersNumber):
                self.matrix[(i,j)] = 1.0/self.playersNumber
                
    def reset(self):
        pass
        
        
    def swap(self, cards):
        pass
        
    def reveal(self, player, card):
        pass
        
    def belief(self):
        return self.matrix

class justReveal:
    def __init__(self, playersNumber):
        self.playersNumber = playersNumber
        self.matrix = np.zeros((self.playersNumber, self.playersNumber))
        for i in range(self.playersNumber):
            for j in range(self.playersNumber):
                self.matrix[(i,j)] = 1.0/self.playersNumber
                
    def reset(self):
        self.matrix = np.zeros((self.playersNumber, self.playersNumber))
        for i in range(self.playersNumber):
            for j in range(self.playersNumber):
                self.matrix[(i,j)] = 1.0/self.playersNumber
        
        
    def swap(self, cards):
        pass
        
    def reveal(self, player, card):
        tempMatrix = np.zeros((self.playersNumber, self.playersNumber))
        tempMatrix[(player,card)] = 1
        for i in range(self.playersNumber):
            if(i != player):
                for j in range(self.playersNumber):
                    if(j != card):
                        tempMatrix[(i,j)] = 1.0/(self.playersNumber-1)
        self.matrix = tempMatrix
        
    def belief(self):
        return self.matrix
        


# turning superposition into predictor classes

class predictorExpectSwap(superpositionOpt):
    def swap(self, cards):
        super().swap(cards, 0.8)

class predictorStandard(superpositionOpt):
    def swap(self, cards):
        super().swap(cards, 0.5)
        
class predictorConstantSwapProbability(superpositionOpt):
# this predictor assumes constant probability of a swap, probability is set at initialisation
    def __init__(self, playersNumber, swapProbability):
        super().__init__(playersNumber)
        self.swapProbability = swapProbability
        
    def swap(self, cards):
        super().swap(cards, self.swapProbability)
   
### test of predictor
cards = predictorConstantSwapProbability(playersNumber,0.7)

cards.swap([0,1])
cards.swap([0,3])
cards.swap([4,6])
cards.reveal(3, 3)
cards.swap([5,6])
print(cards.belief())

### here comes
### enviornament for testing prediction accuracy

# you can test your predictor, just add it to predictors list and add its name to Names
# your predictor has to have methods: reset(), swap(cards), reveal(player, card) and method belief() that returns belief matrix
# bear in mind that predictor method swap doesn't take subjective probability as an argument as it will not be supplemented by testing enviornament

predictors = [baseline(playersNumber), justReveal(playersNumber), predictorConstantSwapProbability(playersNumber, 0.3),
              predictorConstantSwapProbability(playersNumber, 0.5), predictorConstantSwapProbability(playersNumber, 0.8)]
names = ["baseline", "justReveal", "swap 0.3", "swap 0.5", "swap 0.8"]

predictorNames = {}

#mostProbable = []
brierScore = {}
#logLossScore = []

for index, pred in enumerate(predictors):
    predictorNames[pred] = names[index]
    brierScore[pred] = 0

numberOfGames = 20
gameLength = 50
swapOccurence = 0.75 #else reveal random card
swapHappensProbability = 0.5


for i in range(numberOfGames):
    
    
    for pred in predictors:
        pred.reset()
    
    permutation = list(range(playersNumber))
    swaps = []
    for j in range(gameLength):
        #choose action
        if(random.random() < swapOccurence):
            cardA = random.randint(0, playersNumber-1)
            nextCard = random.randint(1, playersNumber-1)
            cardB = (cardA + nextCard) % playersNumber #so card is not swapped with itself
            action = {"type" : "swap", "cards" : [cardA, cardB]}
            if(random.random() < swapHappensProbability):
                temp = permutation[cardA]
                permutation[cardA] = permutation[cardB]
                permutation[cardB] = temp
                swaps.append((cardA, cardB))
        else:
            player = random.randint(0, playersNumber-1)
            action = {"type" : "reveal", "player" : player}
            
        #update predictors
        if(action["type"] == "swap"):
            for pred in predictors:
                pred.swap(action["cards"])
        elif(action["type"] == "reveal"):
            for pred in predictors:
                if(pred.reveal(action["player"], permutation[action["player"]]) == "error"):
                    print(permutation, action["player"], permutation[action["player"]])
                    print(swaps)
                    permutation2 = list(range(playersNumber))
                    for s in swaps:
                        temp = permutation2[s[0]]
                        permutation2[s[0]] = permutation2[s[1]]
                        permutation2[s[1]] = temp
                    print(permutation2)
        
        #gather scores
        #Brier score
        for pred in predictors:
            stepScore = 0
            belief = pred.belief()
            for k in range(playersNumber):
                for l in range(playersNumber):
                    if(permutation[k] == l):
                        stepScore += (1-belief[k][l])**2
                    else:
                        stepScore += belief[k][l]**2
            brierScore[pred] += stepScore
        
for pred in predictors:
    print(predictorNames[pred], brierScore[pred]/numberOfGames/gameLength)
