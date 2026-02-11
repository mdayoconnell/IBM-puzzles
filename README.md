This is Micah's code for solving IBM's Ponder This problem for February 2026

His solution was obtained from backgammon_strategy12.py for a d2, and 
backgammon_solver123456.py for a d6. 

This method assumes a finite board, and uses a dynamic programming approach to maximize the 
expected value of moves according to the rules. Since the problem uses an infinite board, 
better and better estimates for EV(rolls) can be obtained by increasing p_max the number of 
positions. Around p_max > 100, the tail of the probability distribution (chance that p > p_max) becomes
so small that we achieve our desired 6 decimals of precision

This method gives a slight improvement over the "simple" strategy that relies only on rolling doubles.
Included is a "strategy oracle" - show_strategy.py that takes the optimal DP solution and shows where
improvements on the "simple" solution are found.

MonteCarlo method for d2 and d6 are also included (at this stage, much slower than the exact DP solution.)


