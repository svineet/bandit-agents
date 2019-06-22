Outline for bandit experiments
==============================

You walk into a bar. There is a slot machine.
Now your job is to learn how it behaves given 10000 lever pulls

After that you will be given 100 lever pulls to show what you've learned 
Highest reward policy wins.

Let us try out some policies

Sample average with decaying exploration epsilon
================================================

P(event)   explore  exploit
event      1-eps    eps
explore -> Sample a random action from given action space and try it out
           Maintain a running average of reward received for each action
exploit -> Just press the lever that gives max (mean reward estimate)

Every `n_change` or so iterations we will decay eps by factor multiplication
0.99

