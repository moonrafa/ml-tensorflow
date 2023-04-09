import tensorflow as tf
import tensorflow_probability as tfp

# 0 = cold days, 1 = hot days
# first day has 80% chance of being cold
# a cold day has 30% chance of being followed by a hot day
# a hot day has 20% chance of being followed by a cold day
# temperature - mean=0 standard deviation=5 on a cold day
# temperature - mean=15 standard deviation=10 on a hot day

tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]])
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

model = tfd.HiddenMarkovModel(initial_distribution=initial_distribution, transition_distribution=transition_distribution, observation_distribution=observation_distribution, num_steps=7)

mean = model.mean()

with tf.compat.v1.Session() as sess:
    print(mean.numpy())