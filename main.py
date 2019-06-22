import numpy as np

from bandit import SimpleBanditEnvironment


class BanditAgent:
    def __init__(self, num_levers, epsilon):
        self.reward_sum = np.zeros(num_levers)
        self.n_taken = np.zeros(num_levers)
        self.eps = epsilon

    def get_action(self, env, test=False):
        choice = np.random.uniform()
        a = None
        if not test and choice <= self.eps:
            # Explore
            a = np.random.choice(env.get_action_space())
        else:
            # Exploit only, when testing
            a = np.argmax(self.reward_sum/self.n_taken)

        return int(a)

    def train(self, env, iteration):
        trew = 0
        for i in range(iteration):
            print("Episode", i)

            action = self.get_action(env)
            print("Acted with: ", action, flush=True)

            reward = env.step(action)
            print("Reward received: ", reward)

            self.n_taken[action] += 1
            self.reward_sum[action] += reward
            self.eps *= 0.95 # Keep reducing exploration tendency
            trew += reward

            print("Mean rewards for all actions")
            print(self.reward_sum/self.n_taken)
            print("-"*80)

        return trew

    def test(self, env, iteration):
        trew = 0
        for i in range(iteration):
            action = self.get_action(env, test=True)
            reward = env.step(action)

            trew += reward

        return trew


if __name__ == '__main__':
    NUM_LEVERS = 10
    NUM_ITER = 10000
    NUM_TEST = 100

    env = SimpleBanditEnvironment(NUM_LEVERS)

    agent = BanditAgent(NUM_LEVERS, 0.5)
    agent.train(env, NUM_ITER)

    print("Total reward during training: ", env.total_reward)
    print("Mean reward during training: ", env.total_reward/NUM_ITER)

    reward = agent.test(env, NUM_TEST)
    print("Reward in testing phase is: ", reward)
    print("Mean reward in testing phase is:", reward/NUM_TEST)

