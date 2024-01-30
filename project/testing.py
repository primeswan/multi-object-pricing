import numpy as np
from config import num_features, threshold
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def test_model(model, test_data, test_labels):
    result = model.evaluate(test_data, test_labels)
    return result


# For getting the optimal menu and plotting graph
rectangles_to_plot = []

def is_unique(entry, existing_entries):
    for existing_entry in existing_entries:
        column_diff = np.abs(entry - existing_entry)
        if np.all(column_diff < threshold):
            return False, existing_entry
    return True, None


def get_menu(model, test_data):
    # Extracting weights and biases from trained_model and concatenating
    dense_weights, dense_biases = model.dense_layer.get_weights()
    non_trainable_weights, non_trainable_biases = model.non_trainable_layer.get_weights()
    concatenated_weights = np.concatenate([dense_weights, non_trainable_weights], axis=-1)
    concatenated_biases = np.concatenate([dense_biases, non_trainable_biases], axis=-1)
    combined_weights_biases = np.concatenate([concatenated_weights.T, concatenated_biases.reshape(-1, 1)], axis=-1)

    unique_wb = np.empty((0, num_features + 1))

    for type in test_data:
        utilities = np.dot(combined_weights_biases[:, :-1], type) - combined_weights_biases[:, -1]
        max_utility_index = np.argmax(utilities)
        optimal_allocation = combined_weights_biases[max_utility_index].reshape(1, -1)
        is_entry_unique, existing_entry = is_unique(optimal_allocation, unique_wb)
        if is_entry_unique:
            unique_wb = np.vstack((unique_wb, optimal_allocation))
        else:
            row_number = np.where((unique_wb == existing_entry).all(axis=1))[0][0]
            rect = Rectangle(type, 1e-2, 1e-2, color=plt.cm.tab10(row_number))
            rectangles_to_plot.append(rect)

    return(unique_wb)


def plot_outcomes():
    fig, ax = plt.subplots()
    for rect in rectangles_to_plot:
        ax.add_patch(rect)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    plt.show()