import matplotlib.pyplot as plt


class ResultVisualization:
    def __init__(self, company_name, y_trues, y_preds, metric_history):
        self.company_name = company_name
        self.y_trues = y_trues
        self.y_preds = y_preds
        self.metric_history = metric_history

    def plot_results(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.grid(alpha=0.75)
        ax.plot(self.y_trues, lw=3, color='#2ecc71', alpha=0.8, label='Ground truth')
        ax.plot(self.y_preds, lw=3, color='#e74c3c', alpha=0.8, label='Prediction')
        ax.legend()
        ax.set_title(f"Predicted/True Prices of the {self.company_name} company")
        plt.show()

    def plot_metric_history(self, metric):
        plt.plot(range(len(self.metric_history)), self.metric_history)
        plt.xlabel('Time')
        plt.ylabel(f'{metric} Value')
        plt.title(f'{metric} Metric History')
        plt.show()

