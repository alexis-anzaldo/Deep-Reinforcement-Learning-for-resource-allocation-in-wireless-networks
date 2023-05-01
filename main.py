import numpy as np
from DQN_agent import Agent
from Environment import Env_cellular

# ---------------------Environment params / DQN params
'''
Envrionment from Power Allocation in Multi-User Cellular Networks with Deep Q Learning Approach
DOI: 10.1109/ICC.2019.8761431
https://github.com/mengxiaomao
'''

'System Parameters ------------------------------------------------------------------------------'
n_x = 4  # BS on axis x
n_y = 4  # BS on axis y
maxM = 3  # Number of UE at each cell
min_dis = 0.01  # UE minimum distance between UE-BS
max_dis = 1.0  # BS minimum distance between UE-BS
max_p = 38.  # BS maximum transmission power (in DB)
p_n = -114.  # Noise power in dBm
power_num = 10  # Number power levels available (Action space)
'--------------------------------------------------------------------------------------------------'

'Requirement for modeling de Jakes Model -------------------------------------------------------------------'
fd = 10  # Maximum Doppler Frequency ( Reger to Eq. 2)
Ts = 20e-3  # Time intervals between intervals (Refer to Eq. 2)
'------------------------------------------------------------------------------------------------------------'

'Considerations ----------------- ----------------------------------------------------------------------------'
L = 3  # Represents the number of clusters to consider as adjacent (BSs L1=7BS, L2=19BS, L3=37BS)
C = 16  # (Ic) Number of interferers taked account for the localized reward (48 input state neurons)
'-------------------------------------------------------------------------------------------------------------'

# ------------------------------------- Other params
ExpTag = 'FIFO10K'
max_reward = 0

'DQN parameters -----------------------------------------------------------------------------------------------'
Gamma = 0.5  # Discount factor
epsilon = 0.9  # Initial epsilon value
eps_min = 0.01  # Final Epsilon value
batch_size = 256  # Batch Size
lr = 0.001  # Learning Rate
mem_size = 50000  # Memory size
'--------------------------------------------------------------------------------------------------------------'

'---------------------------------------'
train_interval = 10  # Training interval
interval = 500  # Result print interval
replace = 100  # Update target network interval
Ns = 30001  # Number of intervals per episodes
test_intervals = 500  # Test interval
episodes = 1  # Episodes (Each episode randomly deploy UEs with new propagation conditions)
instances = 10  # Instances to average the results
source_seed = 0  # Fixed seed to intialize environment (for replicability)
'---------------------------------------'

save_testing = True  # Turn on to evaluate training and testing performance
saving = False  # Data {training performance, testing permormance, ...}
save_model = False  # Save trained model
load_model = False  # Load model for additional training (parameter Transfer Learning)

filename = ''
Source_Name = ''
chkpt_dir = ''


def Train(env, env_testing, agent, episodes, Ns, interval, max_reward, instance, source_seed):
    Nbs = env.M  # This is the number of agents( i.e. links between BS-UE)
    'Initialization --------------------------------------------------------------------------'
    action = np.zeros(Nbs, dtype=np.int32)
    test_action = np.zeros(Nbs, dtype=np.int32)
    terminal = Ns - 2
    terminal2 = Ns - 2 - interval
    'available Power (According to LTE, this is the minimum power of a Picocell)'
    av_pow = env.get_power_set(5)  # 5 (in dB) is the minimum transmission power
    env.set_Ns(Ns)  # Set the number of intervals for training environment
    env_testing.set_Ns(Ns)  # Set the number of intervals for testing environment
    Rate_dqn_list = list()
    Reward_dqn_list = list()
    Test_Rate_dqn_list = list()
    Test_Reward_dqn_list = list()

    if load_model:
        agent.load_models()
    '--------------------------------------------------------------------------------------------'

    for k in range(1, episodes + 1):
        state, _, _ = env.reset(seed=source_seed)
        state_testing, _, _ = env_testing.reset(seed=source_seed)

        # np.random.seed(instance)
        '-----------------------------------------------------------------------------------------------'
        for i in range(int(Ns) - 1):
            random_epsilon = np.random.random((Nbs))  # Individual Exploration

            for i_agent_BS in np.arange(Nbs):
                action[i_agent_BS] = agent.choose_action(state[i_agent_BS, :], random_epsilon[i_agent_BS])

            'For Testing -----------------------------------------------------------------------------'
            if save_testing:
                # if (i % 10 == 0): # Testing Cada 10 intervalos de tiempo
                for i_agent_BS in np.arange(Nbs):
                    test_action[i_agent_BS] = agent.choose_action_test(state_testing[i_agent_BS, :])
                new_state_testing, _, Test_reward, Test_sumrate = env_testing.step(av_pow[test_action])
                Test_Rate_dqn_list.append(Test_sumrate)  # Vector of mean rates for interval
                Test_Reward_dqn_list.append(np.mean(Test_reward))
                state_testing = np.copy(new_state_testing)
            'For Testing -----------------------------------------------------------------------------'

            new_state, _, reward, sumrate = env.step(av_pow[action])

            for i_agent_BS in np.arange(Nbs):
                agent.store_transition(state[i_agent_BS, :], np.array([action[i_agent_BS]]),
                                       np.array([reward[i_agent_BS]]), new_state[i_agent_BS, :])

            state = np.copy(new_state)

            Rate_dqn_list.append(sumrate)  # Vector of mean rates for interval
            Reward_dqn_list.append(np.mean(reward))

            if i % train_interval == 0:
                agent.learn()

            agent.decay_epsfix(int(0.5 * (Ns - 1)), i)

            if (i % interval == 0):
                reward = np.mean(Reward_dqn_list[-100:])
                test_reward = np.mean(Test_Reward_dqn_list[-10:])
                Average_rate = np.mean(Rate_dqn_list[-100:])
                Test_average_rate = np.mean(Test_Rate_dqn_list[-10:])

                if (i > terminal2) and save_model:
                    agent.save_models()
                print(
                    "Episode(train):%d     interval:%d     Rate: %.3f     Test_Rate: %.3f     Reward: %.3f     Test_Reward:%.3f    Epsilon: %.4f"
                    % (k, i, Average_rate, Test_average_rate, reward, test_reward, agent.epsilon))

    return Rate_dqn_list, Reward_dqn_list, Test_Rate_dqn_list, Test_Reward_dqn_list


Rates_matrix = np.zeros((instances, episodes * (Ns - 1)))
Reward_matrix = np.zeros((instances, episodes * (Ns - 1)))
Test_Rates_matrix = np.zeros((instances, episodes * (Ns - 1)))
Test_Rewards_matrix = np.zeros((instances, episodes * (Ns - 1)))

x_axis = np.arange(0, episodes * (Ns - 1))


env = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)
env_testing = Env_cellular(fd, Ts, n_x, n_y, L, C, maxM, min_dis, max_dis, max_p, p_n, power_num)

max_reward = 0
for i in np.arange(instances):
    q_eval_name = Source_Name + '_' + str(i) + '_eval'
    q_next_name = Source_Name + '_' + str(i) + '_next'

    agent = Agent(gamma=Gamma, epsilon=epsilon, lr=lr, input_dims=[env.state_num], n_actions=power_num,
                  mem_size=mem_size, eps_min=eps_min, batch_size=batch_size, replace=replace,
                  chkpt_dir=chkpt_dir, q_eval_name=q_eval_name, q_next_name=q_next_name, instance=i)

    X_Rate, X_Reward, Y_Rate, Y_Reward = Train(env=env, env_testing=env_testing, agent=agent, episodes=episodes,
                                               Ns=Ns, interval=interval, max_reward=max_reward, instance=i,
                                               source_seed=source_seed)

    Rates_matrix[i, :] = X_Rate
    Reward_matrix[i, :] = X_Reward
    if save_testing:
        Test_Rates_matrix[i, :] = Y_Rate
        Test_Rewards_matrix[i, :] = Y_Reward

if saving:
    np.savez(filename, Training_Rates=Rates_matrix, Training_Rew=Reward_matrix,
             Test_Rates=Test_Rates_matrix, Test_Rew=Test_Rewards_matrix)


import matplotlib.pyplot as plt

x_axis_1 = np.arange(0, len(Rates_matrix[0]))

plt.plot(x_axis_1, np.mean(Rates_matrix, axis=0), label='Training', color='tab:green', linewidth=1.0,
         linestyle='dashed')
plt.fill_between(x_axis_1, np.max(Rates_matrix, axis=0), np.min(Rates_matrix, axis=0), alpha=.3, color='tab:green')
if save_testing:
    plt.plot(x_axis_1, np.mean(Test_Rates_matrix, axis=0), label='Testing', color='tab:orange', linewidth=2.0,
             linestyle='dashed')
    plt.fill_between(x_axis_1, np.max(Test_Rates_matrix, axis=0), np.min(Test_Rates_matrix, axis=0), alpha=.3,
                     color='tab:orange')
plt.grid()
plt.legend()
plt.ylabel('average spectral efficiency (bps/Hz)')
plt.xlabel('Time intervals')
plt.title('Network Performance (Exploration with exponential decay :  0.9 -> 0.001)')
plt.show()

plt.plot(x_axis_1, np.mean(Reward_matrix, axis=0), label='Training', color='tab:blue', linewidth=1.0,
         linestyle='dashed')
plt.fill_between(x_axis_1, np.max(Reward_matrix, axis=0), np.min(Reward_matrix, axis=0), alpha=.3, color='tab:blue')
if save_testing:
    plt.plot(x_axis_1, np.mean(Test_Rewards_matrix, axis=0), label='Testing', color='tab:orange', linewidth=2.0,
             linestyle='dashed')
    plt.fill_between(x_axis_1, np.max(Test_Rewards_matrix, axis=0), np.min(Test_Rewards_matrix, axis=0), alpha=.3,
                     color='tab:orange')
plt.grid()
plt.legend()
plt.ylabel('Reward Function')
plt.xlabel('Time intervals')
plt.title('Model Performance (Exploration with exponential decay :  0.9 -> 0.001) ')
plt.show()

def moving_average(a, n=50):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

a = 50
Training = moving_average(np.mean(Rates_matrix, axis=0),a)
Testing = moving_average(np.mean(Test_Rates_matrix, axis=0),a)
x_axis = np.arange(0, len(Training))
plt.plot(x_axis, Training, label='Training', color='tab:green', linewidth=1.0,
         linestyle='dashed')
if save_testing:
    plt.plot(x_axis, Testing, label='Testing', color='tab:orange', linewidth=2.0,
             linestyle='dashed')
plt.grid()
plt.legend()
plt.ylabel('average spectral efficiency (bps/Hz)')
plt.xlabel('Time intervals')
plt.title('Network Performance (Exploration with exponential decay :  0.9 -> 0.001)')
plt.show()