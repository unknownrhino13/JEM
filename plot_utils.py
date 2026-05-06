import matplotlib.pyplot as plt
import pickle

def plot_losses(losses_arrays, args, fold_save_dir):
    max_length = max((len(values) for values in losses_arrays.values() if values), default=0)

    steps = list(range(max_length))

    metric_specs = [
        ("p_x_loss", "Steps", "Loss Magnitude", "P(x) Loss Over Steps"),
        ("p_y_given_x_loss", "Steps", "Loss Magnitude", "Classification Loss Over Steps"),
        ("fid", "Steps", "FID Score", "FID Score Over Steps"),
        ("acc", "Steps", "Accuracy", "Accuracy Over Steps"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    plot_index = 0

    def plot_data(ax, key, values, xlabel, ylabel, title):
        ax.plot(steps[:len(values)], values, label=key)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    for key, xlabel, ylabel, title in metric_specs:
        if key in losses_arrays and losses_arrays[key]:
            plot_data(axes[plot_index], key, losses_arrays[key], xlabel, ylabel, title)
            plot_index += 1

    for i in range(plot_index, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(f'{fold_save_dir}/combined_plots_p_{args.p}.png')
    plt.close()

    with open(f'{fold_save_dir}/losses_arrays.pkl', 'wb') as pickle_file:
        pickle.dump(losses_arrays, pickle_file)
