# policy function Qpi(s, a), state and action
# Q*(s, a) = reward and return maximized by the old value of s, a
# observation is an array of 4 floats:
# the position and velocity of the cart
# the angular position and velocity of the pole
# reward is a scalar float value
# action is a scalar integer with only two possible values:
# 0 — "move left"
# 1 — "move right"
from __future__ import absolute_import, division, print_function
import gym
import numpy as np
import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common



env_name = 'CartPole-v0'
env = suite_gym.load(env_name)
env.reset()

# The environment’s step function returns exactly what we need.
# In fact, step returns 4 values: observation(object),
# reward(float), done(boolean), info(dict)

print('Observation Spec:' + env.time_step_spec().observation)
print('Reward Spec:' + env.time_step_spec().reward)
print('Action Spec:' + env.action_spec())
time_step = env.reset()
print('Time step:' + time_step)
action = np.array(1, dtype=np.int32)
next_time_step = env.step(action)
print('Next time step:' + next_time_step)

train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

# Use tf_agents.networks.q_network to create a QNetwork,
# passing in the observation_spec, action_spec, and
# a tuple describing the number and size of the model's hidden layers.
fc_layer_params = (100,)
q_net = q_network.QNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)

# A DQN agent is a value-based reinforcement learning agent that trains
# the object to estimate the return or future rewards.
# DQN is a variant of Q-learning.

learning_rate = 1e-3  # int
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
train_step_counter = tf.Variable(0)
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)
agent.initialize()

# Agents contain two policies:
# agent.policy — The main policy that is used for evaluation and deployment.
# agent.collect_policy — A second policy that is used for data collection.
# tf_agents.policies.random_tf_policy is used to create a policy which will
# randomly select an action for each time_step.
eval_policy = agent.policy
collect_policy = agent.collect_policy
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

# action — the action to be taken (in this case, 0 or 1)
# state — used for stateful (that is, RNN(recurrent neural net)-based) policies
# info — auxiliary data, such as log probabilities of actions
env = tf_py_environment.TFPyEnvironment(
    suite_gym.load('CartPole-v0'))
time_step = env.reset()
random_policy.action(time_step)


# The return is the sum of rewards obtained
# while running a policy in an environment for an episode.
# Several episodes are run, creating an average return.

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

num_eval_episodes = 10  # int
compute_avg_return(eval_env, random_policy, num_eval_episodes)


replay_buffer_max_length = 100000  # int
# The replay buffer keeps track of data collected from the environment.
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)
# For most agents, collect_data_spec is a named tuple
# called Trajectory, containing the specs for
# observations, actions, rewards, and other items.
agent.collect_data_spec
agent.collect_data_spec._fields

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    buffer.add_batch(traj)

initial_collect_steps = 100  # int

def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)

collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

# iter(replay_buffer.as_dataset()).next()

# The agent needs access to the replay buffer.
# This is provided by creating an iterable tf.data.Dataset pipeline
# which will feed data to the agent. Each row of the replay buffer only stores a
# single observation step. But since the DQN Agent needs both the current and
# next observation to compute the loss, the dataset pipeline will sample two
# adjacent rows for each item in the batch (num_steps=2).

batch_size = 64  # int
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)
dataset
iterator = iter(dataset)
print(iterator)

# Two things must happen during the training loop:
# collect data from the environment
# use that data to train the agent's neural network(s)
# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

num_iterations = 20000  # int
collect_steps_per_iteration = 1  # int
log_interval = 200  # int
eval_interval = 1000  # int

# The main training implementation of the code that calls methods from above:
for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)

    
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        # action = agent.predict(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()

# observation = [position of cartpole, angle of the cartpole]
# graph_action = agent.policy.predict(graph_observation)
# action = graph_action2action(graph_Action)
# env.step(action)
# graph_action2action - function to condense graph into a vector
