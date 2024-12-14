import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
import tkinter as tk
from tkinter import ttk

class ClusteringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clustering App")

        # Variables to store user-entered data
        self.x_data = []
        self.y_data = []

        # GUI Components
        self.method_var = tk.StringVar()
        self.method_var.set("KMeans")

        self.method_label = ttk.Label(root, text="Select clustering method:")
        self.method_label.pack(pady=5)

        self.method_menu = ttk.Combobox(root, textvariable=self.method_var, values=["KMeans", "Hierarchical"])
        self.method_menu.pack(pady=5)

        self.data_entry_label = ttk.Label(root, text="Enter data points (up to 10):")
        self.data_entry_label.pack(pady=5)

        self.x_label = ttk.Label(root, text="X Data:")
        self.x_label.pack(pady=5)

        self.x_entry = ttk.Entry(root, width=30)
        self.x_entry.pack(pady=5)

        self.y_label = ttk.Label(root, text="Y Data:")
        self.y_label.pack(pady=5)

        self.y_entry = ttk.Entry(root, width=30)
        self.y_entry.pack(pady=5)

        self.cluster_button = ttk.Button(root, text="Cluster", command=self.cluster_data)
        self.cluster_button.pack(pady=10)

        self.result_label = ttk.Label(root, text="")
        self.result_label.pack(pady=10)

    def cluster_data(self):
        method = self.method_var.get()

        x_input = self.x_entry.get()
        y_input = self.y_entry.get()

        try:
            x_data = [float(x) for x in x_input.split(',')[:10]]
            y_data = [float(y) for y in y_input.split(',')[:10]]

            self.x_data = np.array(x_data)
            self.y_data = np.array(y_data)

            if method == "KMeans":
                self.kmeans_clustering()
            elif method == "Hierarchical":
                self.hierarchical_clustering()

        except ValueError:
            self.result_label.config(text="Invalid input. Please enter numeric values.")

    def kmeans_clustering(self):
        # Combine x and y data into a 2D array
        data = np.column_stack((self.x_data, self.y_data))

        # Perform KMeans clustering with three clusters
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(data)
        labels = kmeans.labels_

        # Plot the clustered data
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='X', s=200,
                    label='Centroids')
        plt.title('KMeans Clustering')
        plt.xlabel('X Data')
        plt.ylabel('Y Data')
        plt.legend()
        plt.show()

    def hierarchical_clustering(self):
        # Combine x and y data into a 2D array
        data = np.column_stack((self.x_data, self.y_data))

        # Perform hierarchical clustering with single linkage
        linkage_matrix = linkage(data, method='single')

        # Plot the dendrogram
        plt.figure()
        dendrogram(linkage_matrix)
        plt.title('Hierarchical Clustering')
        plt.xlabel('Data Points')
        plt.ylabel('Distance')
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = ClusteringApp(root)
    root.mainloop()
