import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def segement_image_with_kmeans(img, num_colors):
    X = img.reshape(-1,3)
    kmeans = KMeans(n_clusters=num_colors, random_state=42).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    segmented_img = centers[labels.flatten()]
    segmented_img = segmented_img.reshape(img.shape)
    return segmented_img

def plot_segmented_images(original_img):
    clusters = [2,3,4]
    fig, axs = plt.subplots(1, len(clusters)+1, figsize=(18,6))
    axs[0].imshow(original_img)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    for i in range(1,len(clusters)+1):
        axs[i].imshow(segement_image_with_kmeans(original_img, clusters[i-1]).astype(np.uint8))
        axs[i].set_title('Segmented Image' +'\n' + 'to ' + str(clusters[i-1]) + 'colors')
        axs[i].axis('off')

    fig.tight_layout()

images = ['tree.png']
for img in images:
    img = plt.imread(img,0)
    plot_segmented_images(img)

plt.show()
