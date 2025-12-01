import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics



class Evaluator:
    """
    A class for evaluating a biometric system's performance.
    """

    def __init__(self, 
                 num_thresholds, 
                 genuine_scores, 
                 impostor_scores, 
                 plot_title, 
                 epsilon=1e-12):
        """
        Initialize the Evaluator object.

        Parameters:
        - num_thresholds (int): Number of thresholds to evaluate.
        - genuine_scores (array-like): Genuine scores for evaluation.
        - impostor_scores (array-like): Impostor scores for evaluation.
        - plot_title (str): Title for the evaluation plots.
        - epsilon (float): A small value to prevent division by zero.
        """
        self.num_thresholds = num_thresholds
        self.thresholds = np.linspace(-1.1, 1.1, num_thresholds)
        self.genuine_scores = genuine_scores
        self.impostor_scores = impostor_scores
        self.plot_title = plot_title
        self.epsilon = epsilon

    def get_dprime(self):
        """
        Calculate the d' (d-prime) metric.

        Returns:
        - float: The calculated d' value.
        """
        self.genuine_mean = np.mean(self.genuine_scores)
        self.imposter_mean = np.mean(self.impostor_scores)
        self.genuine_std = np.std(self.genuine_scores)
        self.impostor_std = np.std(self.impostor_scores)
        x = abs(np.mean(self.genuine_scores) - np.mean(self.impostor_scores))
        y = np.sqrt(0.5 * (np.var(self.genuine_scores) + np.var(self.impostor_scores)))
        return x / (y + self.epsilon)

    def plot_score_distribution(self):
        """
        Plot the distribution of genuine and impostor scores.
        """
        plt.figure(figsize=(9,9))
        
        # Plot the histogram for genuine scores
        plt.hist(
            self.genuine_scores,
            bins=50,
            color='green',
            lw=2,
            histtype='step',
            label='Genuine'
        )
        
        # Plot the histogram for impostor scores
        plt.hist(
            self.impostor_scores,
            bins=50,
            color='red',
            lw=2,
            histtype='step',
            label='Impostor'
        )
        
        # Set the x-axis limit to ensure the histogram fits within the correct range
        plt.xlim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        
        # Step 6: Add legend to the upper left corner with a specified font size
        plt.legend(
            # loc: Specify the location for the legend (e.g., 'upper left')
            loc='upper left', 
            # fontsize: Set the font size for the legend
            fontsize=10
        )
        
        # Step 7: Set x and y-axis labels with specified font size and weight
        plt.xlabel(
            # Provide the x-axis label
            'Score Value',
            # fontsize: Set the font size for the x-axis label
            fontsize=12,
            # weight: Set the font weight for the x-axis label
            weight='bold'
        )
        
        plt.ylabel(
            # Provide the y-axis label
            'Score Frequency',
            # fontsize: Set the font size for the y-axis label
            fontsize=12,
            # weight: Set the font weight for the y-axis label
            weight='bold'
        )
        
        # Step 8: Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Step 9: Set font size for x and y-axis ticks
        plt.xticks(
            # fontsize: Set the font size for x-axis ticks
            fontsize=10
        )
        
        plt.yticks(
            # fontsize: Set the font size for y-axis ticks
            fontsize=10
        )
        
        # Step 10: Add a title to the plot with d-prime value and system title
        plt.title('Score Distribution Plot\nd-prime= %.2f\nSystem %s' % 
                  (self.get_dprime(), 
                   self.plot_title),
                  fontsize=15,
                  weight='bold')
        
        # Save the figure
        plt.savefig('score_distribution_plot_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")
        
        # Close the figure to free up resources (no plt.show() to avoid blocking)
        plt.close()

        return

    def get_EER(self, FPR, FNR):
        """
        Calculate the Equal Error Rate (EER).
    
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - FNR (list or array-like): False Negative Rate values.
    
        Returns:
        - float: Equal Error Rate (EER).
        """
        EER = 0
        
        # Add code here to compute the EER
        diff = np.abs(FPR - FNR)
        idx = np.argmin(diff)
        EER = (FPR[idx] + FNR[idx]) / 2
        EER_threshold = self.thresholds[idx]
        
        return EER, EER_threshold

    def plot_det_curve(self, FPR, FNR):
        """
        Plot the Detection Error Tradeoff (DET) curve.
        Parameters:
         - FPR (list or array-like): False Positive Rate values.
         - FNR (list or array-like): False Negative Rate values.
        """
        
        # Calculate the Equal Error Rate (EER) using the get_EER method
        EER, EER_threshold = self.get_EER(FPR, FNR)
        
        # Create a new figure for plotting
        plt.figure(figsize=(9,9))
        
        # Plot the Detection Error Tradeoff Curve
        plt.plot(
            # FPR values on the x-axis
            FPR,
            # FNR values on the y-axis
            FNR,
            # lw: Set the line width for the curve
            lw=2,
            # color: Set the color for the curve
            color='black'
        )
        
        # Add a text annotation for the EER point on the curve
        # Plot the diagonal line representing random classification
        # Scatter plot to highlight the EER point on the curve

        plt.text(EER + 0.07, EER + 0.07, "EER", style='italic', fontsize=12,
                 bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        plt.scatter([EER], [EER], c="black", s=100)
        
        # Set the x and y-axis limits to ensure the plot fits within the range 
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(
            # color: Set the color of grid lines
            color='gray',
            # linestyle: Choose the line style for grid lines
            linestyle='--',
            # linewidth: Set the width of grid lines
            linewidth=0.5
        )
        
        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set x and y-axis labels with specified font size and weight
        plt.xlabel(
            # 'False Pos. Rate': Set the x-axis label
            'False Positive Rate',
            # fontsize: Set the font size for the x-axis label
            fontsize=12,
            # weight: Set the font weight for the x-axis label
            weight='bold'
        )
        
        plt.ylabel(
            # 'False Neg. Rate': Set the y-axis label
            'False Negative Rate',
            # fontsize: Set the font size for the y-axis label
            fontsize=12,
            # weight: Set the font weight for the y-axis label
            weight='bold'
        )
        
        # Step 11: Add a title to the plot with EER value and system title
        plt.title(
            'Detection Error Tradeoff Curve \nEER = %.5f at t=%.3f\nSystem %s' %
            (EER, EER_threshold, self.plot_title), fontsize=14, weight='bold' 
            # EER: Provide the calculated EER value
            # EER_threshold: Provide the threshold at which the EER occurs
            # self.plot_title: Provide the system title
            # fontsize: Set the font size for the title
            # weight: Set the font weight for the title
        )
        
        # Step 12: Set font size for x and y-axis ticks
        plt.xticks(
            # fontsize: Set the font size for x-axis ticks
            fontsize=10
        )
        
        plt.yticks(
            # fontsize: Set the font size for y-axis ticks
            fontsize=10
        )
        
        # Save the plot as an image file
        plt.savefig(
            'DET_curve_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight"
        )
        
        # Close the plot to free up resources (no plt.show() to avoid blocking)
        plt.close()
    
        return

    def plot_roc_curve(self, FPR, TPR):
        """
        Plot the Receiver Operating Characteristic (ROC) curve.
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - TPR (list or array-like): True Positive Rate values.
        """

        auc_value = metrics.auc(FPR, TPR)
        
        # Create a new figure for the ROC curve
        plt.figure(figsize=(9, 9))
        # Plot the ROC curve using FPR and TPR with specified attributes
        plt.plot(FPR, TPR, lw=2, color='blue')
        plt.plot([0,1], [0,1], '--', color = 'gray', lw=0.5)
        # Set x and y axis limits, add grid, and remove top and right spines\
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        # Set labels for x and y axes, and add a title
        plt.xlabel("False Positive Rate (FPR)", fontsize=14)
        plt.ylabel("True Positive Rate (TPR)", fontsize=14)
        plt.title('ROC Curve = %.5f\nSystem %s' % (auc_value, self.plot_title), fontsize=16, weight='bold')
        # Set font sizes for ticks, x and y labels
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # Save the plot as a PNG file
        plt.savefig('ROC_curve(%s).png' % self.plot_title, dpi=300, bbox_inches='tight')
        
        # Close the figure to free up resources (no plt.show() to avoid blocking)
        plt.close()
 
        return

    def compute_rates(self):

        # Initialize lists for False Positive Rate (FPR), False Negative Rate (FNR), and True Positive Rate (TPR)
        FPR, FNR, TPR = [], [], []

        # Iterate through threshold values and calculate TP, FP, TN, and FN for each threshold
        for t in self.thresholds:
            TP = np.sum(self.genuine_scores >= t)
            FN = np.sum(self.genuine_scores < t)
            FP = np.sum(self.impostor_scores >= t)
            TN = np.sum(self.impostor_scores < t)

        # Append calculated rates to their respective lists
            FPR.append(FP / (FP + TN + self.epsilon))     # Calculate FPR, FNR, and TPR based on the obtained values
            FNR.append(FN / (TP + FN + self.epsilon))
            TPR.append(TP / (TP + FN + self.epsilon))

        # Return the lists of FPR, FNR, and TPR
        return np.array(FPR), np.array(FNR), np.array(TPR)

# NOTE: To run the full system, use main.py instead.
# This file is only for the Evaluator class.

if __name__ == "__main__":
    print("This module provides the Evaluator class.")
    print("To run the full biometric system, use: python main.py")

