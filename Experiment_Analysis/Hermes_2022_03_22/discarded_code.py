

type = orientation_type_of_all_trials[0]
sum_of_successes_in_block = []
number_of_fails_before_correct = []
sum = 0
num_of_fails = 0
num_of_successes = 0
for i, t in enumerate(positions_of_all_trials):
    if t in positions_of_successes:
        sum += 1
        num_of_successes +=1
    else:
        if num_of_successes == 0:
            num_of_fails += 1
    sum_of_successes_in_block.append(sum)
    if orientation_type_of_all_trials[i] != type:
        number_of_fails_before_correct.append(num_of_fails)
        sum = 0
        num_of_fails = 0
        num_of_successes = 0
        type = orientation_type_of_all_trials[i]



# For 3 coefficients
if regression_coefficients.shape[1] == 3:
    plt.rc('font', size=20)
    fig = plt.figure(1)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs=regression_coefficients[:, 0], ys=regression_coefficients[:, 1], zs=regression_coefficients[:, 2],
               c=np.arange(1, 0, -1/len(regression_coefficients[:, 0])), cmap='jet')
    ax.set_xlabel('Left or Right previous Trial')
    ax.set_ylabel('Successful or Failed previous Trial')
    ax.set_zlabel('Trial Orientation')
    ax.set_title('{}'.format(experiment_date[date]))
    fig = plt.figure(2)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs=np.arange(0, 1, 1/len(regression_coefficients[:, 0])), ys=regression_coefficients[:, 0], zs=regression_coefficients[:, 2],
               c=regression_scores, cmap='jet', vmax=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Left or Right previous Trial')
    ax.set_zlabel('Trial Orientation')
    ax.set_title('{}'.format(experiment_date[date]))
    fig = plt.figure(3)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs=np.arange(0, 1, 1 / len(regression_coefficients[:, 0])), ys=regression_coefficients[:, 1],
               zs=regression_coefficients[:, 2],
               c=regression_scores, cmap='jet', vmax=1)
    ax.set_xlabel('Time')
    ax.set_ylabel('Successful or Failed previous Trial')
    ax.set_zlabel('Trial Orientation')
    ax.set_title('{}'.format(experiment_date[date]))


# For 2 coefficients
# Z = Time
if regression_coefficients.shape[1] == 2:
    plt.rc('font', size=20)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    c = number_of_transitions_in_trials_group / (number_of_trials_to_regress - 1)
    c = regression_scores
    ax.scatter(ys=regression_coefficients[:, 0], zs=regression_coefficients[:, 1], xs=np.arange(0, 1, 1/len(regression_coefficients[:, 0])),
               c=c, cmap='jet', vmax=1, s=20)
    ax.set_xlabel('Time')
    ax.set_ylabel('Success Stay')
    ax.set_zlabel('Trial Orientation')
    ax.set_title('{}'.format(experiment_date[date]))
    plt.tight_layout()