import tensorflow as tf
import numpy as np
import random
from training import train_model
from testing import test_model, get_menu, plot_outcomes
from config import train_size, num_features, test_size

##################

# Setting random seeds for reproducibility
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)

##################

# Training data
train_data = np.random.uniform(0, 1, size=(train_size, num_features))
# Train_labels are zeros
train_labels = np.zeros((train_size, 1))

# Training
trained_model = train_model(train_data, train_labels)

###################

# Testing data
test_data = np.random.uniform(0, 1, size=(test_size, num_features))
test_labels = np.zeros((test_size, 1))

# Testing
expected_revenue = test_model(trained_model, test_data, test_labels)
rounded_revenue = np.round(expected_revenue, 4)
print('Expected Revenue:', rounded_revenue)

####################

# Getting the menu, this also prints the outcomes on the type-space
optimal_menu = get_menu(trained_model, test_data)
rounded_menu = np.round(optimal_menu, 2)
print('Optimal Menu:', rounded_menu)


#######################

# Outcomes graph (get the menu before running this)
plot_outcomes()
