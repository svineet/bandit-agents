"""
I am a bachelor's student of Computer Science at BITS Pilani, India.

I have been programming since I was in 7th grade. I have several international and national awards to my name, including Google Code-in for which I was invited to the Googleplex for an all expense paid trip and met Google engineers.

I am a python expert. I am well versed in web development with python as a backend. On the frontend I am well versed with ES6 and ReactJS. I also have experience with quantitative finance and related infrastructure. I have a good hold on algorithms and data structures as well. I have basic experience in Deep Learning and Machine Learning based solutions. Currently learning more about reinforcement learning and its applications.

You can see my public projects and history in open source on my GitHub profile: github.com/svineet

Willing to work on interesting deep learning and finance related projects. Look forward to working with you.
"""
import numpy as np


class SimpleBanditEnvironment:
    def __init__(self, num_levers):
        self.means = np.random.randn(num_levers)*10
        self.stds = np.random.uniform(size=num_levers)*9.5 + 0.5

        self.total_reward = 0

    def get_action_space(self):
        return range(self.means.shape[0])

    def step(self, action):
        """
            Takes
                - action: scalar key from [0, self.num_levers)

            Returns:
                - reward: scalar
        """

        r = np.random.normal(self.means[action], self.stds[action])
        self.total_reward += r

        return r
