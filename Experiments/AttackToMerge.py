import funcs


last_row_handled = -1
true_batch, last_row_handled = funcs.extract_batch(df, 32, last_row_handled)

# Make sure to actually call .copy() as a method
batch = true_batch.copy()

true_label = funcs.predict_LM(batch)
current_label = true_label.copy()

for perturb_iteration in range(num_perturb_iterations):

    # attempt to flip with gaussian perturbation:
    for index, datapoint in batch.iterrows():
        if true_label[index] == current_label[index]:
            temp_dp = funcs.categorical_gaussian_perturb(datapoint.to_frame().T, catfeature_sigma_dict, correlation_dict)
            temp_dp = funcs.numerical_gaussian_perturb(temp_dp.to_frame().T, numfeature_sigma_dict, value_dict)
            batch.iloc[index] = temp_dp

    # check if labels flipped
    current_label = funcs.predict_LM(batch)

    # Reset unchanged points back to original if they haven't flipped
    for index in batch.index:
        if true_label[index] == current_label[index]:
            batch.iloc[index] = true_batch.iloc[index]


for minimize_iteration in range(num_minimize_iteration):









