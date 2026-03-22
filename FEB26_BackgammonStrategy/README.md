The simple solution only allows survival on a double roll. 
Improvements over the simple solution are found by exploiting 3..2 situations, moving one man twice to get a 2..3

backgammon_strategy.py solves this problem using the Bellman equation, solving the EV of rolls from a set of finite
board positions up to p_max. Thus, the game terminates when a blot cannot be avoided, or when a man reaches or exceeds
p_max. By increasing p_max, we search over more and more states, and approach the limit where the board has an 
infinite number of positions. p_max = 200 were necessary to achieve 6 digits of precision. In other words, the probability
mass beyond that mark is < 10e-6

show_strategy.py details the cases where an improvement over the simple solution was found
montecarlo.py was a preliminary attempt that was far too noisy for the precision that the problem required

