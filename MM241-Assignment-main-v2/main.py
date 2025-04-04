import gym_cutting_stock
import gymnasium as gym
from policy import GreedyPolicy, RandomPolicy
# from student_submissions.s2352888.policy2352888 import Policy2352888
from student_submissions.s23.policy2352888_2352818_2353036_2353298_2352738 import Policy2352888_2352818_2353036_2353298_2352738

# Create the environment
env = gym.make(
    "gym_cutting_stock/CuttingStock-v0",
    render_mode="human",  # Comment this line to disable rendering
)
NUM_EPISODES = 1

if __name__ == "__main__":
    # Reset the environment
    observation, info = env.reset(seed=42)

    # Test GreedyPolicy
    gd_policy = GreedyPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = gd_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(info)
            observation, info = env.reset(seed=ep)
            ep += 1

    # Reset the environment
    observation, info = env.reset(seed=42)

    # Test RandomPolicy
    rd_policy = RandomPolicy()
    ep = 0
    while ep < NUM_EPISODES:
        action = rd_policy.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(info)
            observation, info = env.reset(seed=ep)
            ep += 1

######### Chỉnh trong phạm vi này ########################################################################################
    # Test Policy2352888
    observation, info = env.reset(seed=42)
    print(info)# in ra thông tin sau mỗi lần chạy

    policy2352888 = Policy2352888_2352818_2353036_2353298_2352738(policy_id=1)
    for _ in range(200): #chỉnh số lần chạy(để mặc định theo file gốc là 200)
        action = policy2352888.get_action(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        print(info) # in ra tung bước trong quá trình chạy

        if terminated or truncated:
            # print(info)
            observation, info = env.reset()
#################################################################################################


env.close()
