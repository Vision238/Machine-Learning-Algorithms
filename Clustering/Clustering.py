# Vision Aggarwal
# Roll Number_B20171

# importing required modules:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN


# Importing data
data = pd.read_csv("Iris.csv")
# Dropping Class Attribute:
data_redu = data.drop(['Species'], axis=1)
standard = preprocessing.scale(data_redu)
# print(standard)


# Function for finding the purity score of clustered data:
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)

    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum()/np.sum(contingency_matrix)


# Function for  finding Distortion error of clustered data:
def distortion_error(dataset, labels, k):
    data_pca = pd.DataFrame(dataset)
    data_pca["label"] = labels
    # dis  is distortion measure variable:
    dis = 0
    for i in range(k):
        df = data_pca.groupby('label').get_group(i)
        df = np.array(df.drop(["label"], axis=1))
        # Distortion measure of each cluster is calculated separately the added.
        for j in range(len(df)):
            dis += np.linalg.norm(df[j] - np.mean(df, axis=0))
    return dis


# "---------------------------------Question No. 1-------------------------------"
# Plotting eigen Values Vs component graph:
def Question_1():
    # Covariance Matrix:
    t = pd.DataFrame(data_redu).cov()
    eigen_values, eigen_vectors = np.linalg.eig(t)
    plt.bar(['1', '2', '3', '4'], eigen_values, width=0.5, color="#00CDCD")
    plt.xlabel("Components")
    plt.ylabel("EigenValue")
    plt.title("Eigenvalue vs. Components", loc="left")
    # Below Function show exact number above each bar of graph:
    for i in range(len(eigen_values)):
        plt.text(x=i - 0.3, y=(eigen_values[i] + 0.004), s="%.4f" % eigen_values[i], size=9)
    plt.show()


# Performing PCA On dataset:
pca = PCA(n_components=2)
pca.fit(data_redu)
df_pca = pca.fit_transform(data_redu)


def Question_2():
    print()
    print("---------------------------------Question No. 2-------------------------------")
    print()
    # K means number of cluster need to form in the dataset:
    K = 3
    # Initialising,fitting and predicting data points using KMeans:
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(df_pca)
    centers = kmeans.cluster_centers_
    # Plotting each cluster with different color:
    d = plt.scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans.labels_, cmap='rainbow',s=20)
    plt.legend(handles=d.legend_elements()[0], labels=['A', 'b', 'c'])
    # Plotting the Center of each Cluster:
    plt.scatter(centers[:, 0], centers[:, 1], s=80, marker="s", color='#2F4F4F', label="Cluster Centers")
    plt.xlabel("1st Attribute of reduced dataset")
    plt.ylabel("2nd Attribute of reduced dataset")
    plt.title("Clustering by Kmeans on reduced dataset", loc='left')
    plt.legend()
    plt.show()
   
    print("Distortion Error =", "%.3f" % kmeans.inertia_)
    print("Purity of Clustering =", "%.3f" % purity_score(data['Species'], kmeans.labels_))
    print()


def Question_3():
    print()
    print("---------------------------------Question No. 3-------------------------------")
    print()
    # dis_error and purity stores distortion measure and purity score values respectively for different values of K:
    dis_error = []
    purity = []
    # K value:
    k_values = [i for i in range(1, 8)]
    for k in k_values:
        # Initialising,fitting and predicting data points using KMeans:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df_pca)
        dis_error.append(kmeans.inertia_)
        purity.append(round(purity_score(data['Species'], kmeans.labels_), 3))
    # Plotting Distortion error Vs K Value:
    plt.plot(k_values, dis_error, color="#EE3B3B")
    plt.scatter(k_values, dis_error)
    for i in range(len(k_values)):
        plt.text(x=k_values[i], y=dis_error[i], s="%.3f" % (dis_error[i]))
    plt.xlabel("Number of Cluster K")
    plt.ylabel('Value of Distortion Error')
    plt.title("K Value vs Distortion Error Graph", loc='left')
    plt.show()
    
    print("Optimal Number of Clusters is '3'")
    print()
    print("Purity score for different values of K")
    for i in range(len(purity)):
        print("For K = %i" % k_values[i], "purity score = %.3f" % (purity[i]))
    print()


def Question_4():
    print()
    print("---------------------------------Question No. 4-------------------------------")
    print()
    # Initialising, fitting and predicting data points using GMM:
    g = GaussianMixture(n_components=3)
    g.fit(df_pca)
    GMM_prediction = g.predict(df_pca)
    # Plotting each cluster with different colour:
    plt.scatter(df_pca[:, 0], df_pca[:, 1], s=50, c=GMM_prediction, cmap='Accent')
    plt.scatter(g.means_[:, 0], g.means_[:, 1], s=70, color='#CD3333', label="Center of Cluster")
    plt.xlabel("1st Attribute of reduced dataset")
    plt.ylabel("2nd Attribute of reduced dataset")
    plt.title("Data Points After GMM", loc='left')
    plt.legend()
    plt.show()
    
    print("Total Log likelihood", "%.3f" % np.sum(g.score_samples(df_pca)))
    print("Purity of GMM Clustering", "%.3f" % purity_score(data['Species'], GMM_prediction))
    print()


def Question_5():
    # dis_error and purity stores distortion measure and purity score values respectively for different values of K:
    print()
    print("---------------------------------Question No. 5-------------------------------")
    print()
    dis_error = []
    purity = []
    # K value:
    k_values = [i for i in range(1, 8)]
    for k in k_values:
        # Initialising, fitting and predicting data points using GMM:
        g = GaussianMixture(n_components=k)
        g.fit(df_pca)
        GMM_prediction = g.predict(df_pca)
        dis_error.append(np.sum(g.score_samples(df_pca)))
        purity.append(purity_score(data['Species'], GMM_prediction))
    # Plotting Distortion error Vs K Value:
    plt.plot(k_values, dis_error, color="#EE3B3B")
    plt.scatter(k_values, dis_error)
    for i in range(len(k_values)):
        plt.text(x=k_values[i], y=dis_error[i], s="%.3f" % (dis_error[i]))
    plt.xlabel("Number of Cluster K")
    plt.ylabel('Total data log likelihood')
    plt.title("K Value vs total data log likelihood", loc='left')
    plt.show()
    print("Purity score for different values of K")
    
    for i in range(len(purity)):
        print("For K = %i" % k_values[i], "purity score = %.3f" % (purity[i]))
    print()


def Question_6():
    # epsl and min represents  Epsilon and  Min points respectively:
    epsl, min = [1, 1, 4, 4], [4, 10, 4, 10]
    print()
    print("---------------------------------Question No. 6-------------------------------")
    for i in range(4):
        # Initialising, fitting and predicting data points using DBSCAN:
        dbscan_model = DBSCAN(eps=epsl[i], min_samples=min[i]).fit(df_pca)
        DBSCAN_predictions = dbscan_model.labels_
        plt.scatter(df_pca[:, 0], df_pca[:, 1], s=80, c=DBSCAN_predictions, cmap='Accent')
        plt.xlabel("1st Attribute of reduced dataset")
        plt.ylabel("2nd Attribute of reduced dataset")
        plt.title("Clustering Using DBSCAN for 'epsl' {} and 'min' {}".format(epsl[i], min[i]), loc="left")
        plt.show()
        print("purity score for 'epsl' {} and 'min_samples' {} =".format(epsl[i], min[i]),
              "%.3f" % (purity_score(data['Species'], DBSCAN_predictions)))


Question_1()
Question_2()
Question_3()
Question_4()
Question_5()
Question_6()
