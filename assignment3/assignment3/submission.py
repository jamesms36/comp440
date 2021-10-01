import collections, util, math, random

############################################################
# Problem 4.1.1

def computeQ(mdp, V, state, action):
    """
    Return Q(state, action) based on V(state).  Use the properties of the
    provided MDP to access the discount, transition probabilities, etc.
    In particular, MDP.succAndProbReward() will be useful (see util.py for
    documentation).  Note that |V| is a dictionary.  
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    val = 0
    discount = mdp.discount()
    for tup in mdp.succAndProbReward(state, action):
        val += tup[1] * (tup[2] + discount * V[tup[0]])
    return val

############################################################
# Problem 4.1.2

def policyEvaluation(mdp, V, pi, epsilon=0.001):
    """
    Return the value of the policy |pi| up to error tolerance |epsilon|.
    Initialize the computation with |V|.  Note that |V| and |pi| are
    dictionaries.
    """
    # V_old = V
    V_new = {}
    diff = 1000000

    while (diff > epsilon):
        # Computes all the values on the graph
        max_diff = 0
        for s in mdp.states:
            new_val = computeQ(mdp, V, s, pi[s])
            V_new[s] = new_val
            node_diff = abs(V_new[s] - V[s])
            if node_diff > max_diff:
                max_diff = node_diff

        # Compares the new graph to the old graph
        V = dict.copy(V_new)
        diff = max_diff

    return V_new


############################################################
# Problem 4.1.3

def computeOptimalPolicy(mdp, V):
    """
    Return the optimal policy based on V(state).
    You might find it handy to call computeQ().  Note that |V| is a
    dictionary.
    """
    policies = {}
    for state in mdp.states:
        max_policy = None
        max_value = -9999999999
        for action in mdp.actions(state):
            val = computeQ(mdp, V, state, action)
            if val > max_value:
                max_value = val
                max_policy = action

        policies[state] = max_policy

    return policies

############################################################
# Problem 4.1.4

class PolicyIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()

        V = {}
        for state in mdp.states:
            V[state] = 0
            actions = mdp.actions(state)

        pi = computeOptimalPolicy(mdp, V)

        converged = False
        while not converged:

            # Computes all the values on the graph
            converged = True
            for s in mdp.states:
                old_val = V[s]
                new_val = computeQ(mdp, V, s, pi[s])
                V[s] = new_val
                if abs(old_val - new_val) > epsilon:
                    converged = False

            pi = computeOptimalPolicy(mdp, V)


        self.pi = pi
        self.V = V

############################################################
# Problem 4.1.5

class ValueIteration(util.MDPAlgorithm):
    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()

        V = {}
        for state in mdp.states:
            V[state] = 0
            actions = mdp.actions(state)

        pi = computeOptimalPolicy(mdp, V)

        while True:
            V = policyEvaluation(mdp, V, pi, epsilon)
            newPi = computeOptimalPolicy(mdp, V)

            allSame = True
            for state in mdp.states:
                if newPi[state] != pi[state]:
                    allSame = False
                    break

            if allSame:
                break

            pi = dict.copy(newPi)

        self.pi = pi
        self.V = V

############################################################
# Problem 4.1.6

# If you decide 1f is true, prove it in writeup.pdf and put "return None" for
# the code blocks below.  If you decide that 1f is false, construct a
# counterexample by filling out this class and returning an alpha value in
# counterexampleAlpha().
class CounterexampleMDP(util.MDP):
    def __init__(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        print("counterexample")
        # END_YOUR_CODE

    def startState(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return 0
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        if state==0:
            return [1,2]
        else:
            return []
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        if state == 0:
            return [(1, 0.01, 100),
                    (2, 0.99, 10)]
        else:
            return []
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (around 1 line of code expected)
        return 1
        # END_YOUR_CODE

def counterexampleAlpha():
    # BEGIN_YOUR_CODE (around 1 line of code expected)
    return 1000
    # END_YOUR_CODE

def runCounterexample():
    

############################################################
# Problem 4.2.1

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):

        # Takes card:
        #   State: update deckCardCounts, set peekIndex to None, update total card values
        #   Prob: probability that we got this card
        #   Reward: does not change
        if action == 'Take':
            return 0
        # Peeks (only do if peekIndex is None):
        #   State: update peekIndex to be index
        #   Prob: probability we get that index
        #   Reward: decrease by peekcost
        elif action == 'Peek':
            return 0
        # Quits:
        #   State: make the deckCardCounts into None
        #   Prob: 1
        #   Reward: cardValues
        elif action == 'Quit':
            state[2] = None
            return state, 1, self.cardValues

    def discount(self):
        return 1

############################################################
# Problem 4.2.2

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (around 2 lines of code expected)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

