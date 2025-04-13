import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# def plot_line(train, val, path):
#     x = np.array(train)
#     y = np.array(val)
#     size = range(1, len(x) + 1)
#
#     plt.figure(figsize=(10, 8))
#
#     plt.grid(linestyle="-.")
#
#     line_width = 2.0
#     marker_size = 7
#
#     plt.plot(size, x, marker='s', markersize=marker_size, color="blue", label="Train", linewidth=line_width)
#     plt.plot(size, y, marker='X', markersize=marker_size, color="tomato", label="Valid", linewidth=line_width)
#
#     group_labels = [f'{int(i * 20)}%' for i in range(5)]
#
#     x_ticks = np.linspace(min(size), max(size), num=5)
#     plt.xticks(x_ticks, group_labels, fontsize=15)
#
#     y_min, y_max = min(np.min(x), np.min(y)), max(np.max(x), np.max(y))
#     y_ticks = np.linspace(y_min, y_max, num=5)
#     y_labels = [f'{tick:.2f}' for tick in y_ticks]
#     plt.yticks(y_ticks, y_labels, fontsize=15)
#
#     plt.title("Epoch-Loss")
#     plt.xlabel("Epoch", fontsize=15)
#     plt.ylabel("Loss", fontsize=15)
#     plt.xlim(min(size) - 0.5, max(size) + 0.5)
#     plt.ylim(y_min - (y_ticks[-4] - y_ticks[-5]) * 0.5, y_max + (y_ticks[-1] - y_ticks[-2]) * 0.5)
#
#     plt.legend(loc=0, numpoints=1, ncol=2)
#     leg = plt.gca().get_legend()
#     ltext = leg.get_texts()
#     plt.setp(ltext, fontsize=15)
#     plt.tight_layout()
#     plt.savefig(path, format='png')
#     plt.show()

def plot_line(train, val, path):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=list(range(1, len(train) + 1)), y=train, mode='lines+markers', name='train_loss',
                   line=dict(color='blue'),
                   marker=dict(symbol='circle', size=7, color='blue'),
                   showlegend=True))

    fig.add_trace(
        go.Scatter(x=list(range(1, len(val) + 1)), y=val, mode='lines+markers', name='valid_loss',
                   line=dict(color='red'),
                   marker=dict(symbol='square', size=7, color='red'),
                   showlegend=True))

    fig.update_layout(title='Training and Validation Loss',
                      xaxis_title='Epoch',
                      yaxis_title='Loss',
                      width=1200,
                      height=800)

    fig.write_image(path)
