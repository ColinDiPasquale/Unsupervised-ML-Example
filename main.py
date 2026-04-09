import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_images(folder, size=(32, 32)): # Loads all the images
    data = []
    labels = []

    for label in os.listdir(folder):
        path = os.path.join(folder, label)

        if not os.path.isdir(path):
            continue

        for file in os.listdir(path):
            img_path = os.path.join(path, file)

            img = Image.open(img_path).convert('RGB').resize((32, 32)) # Images had transparency initially as well
            img = img.resize(size)

            arr = np.array(img, dtype=np.float32).flatten()
            data.append(arr)
            labels.append(label)

    return np.array(data), np.array(labels)

def predict(test_img, X_pca, y, pca, mean):
    test_centered = test_img - mean
    test_proj = pca.transform([test_centered])[0]

    distances = np.linalg.norm(X_pca - test_proj, axis=1)
    idx = np.argmin(distances)

    return y[idx], distances[idx]

def predict(test_img, X_pca, y, pca, mean):
    test_centered = test_img - mean
    test_proj = pca.transform([test_centered])[0]

    distances = np.linalg.norm(X_pca - test_proj, axis=1)
    idx = np.argmin(distances)

    return y[idx], distances[idx]

X, y = load_images("Blocks")

print("Shape:", X.shape) # Just making sure everything worked
print("Labels:", y[:5])

X_mean = np.mean(X, axis=0) # Getting the mean of the array
X_centered = X - X_mean # Subtracting out the mean

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_centered)

print("Reduced shape:", X_pca.shape)

for i in range(1, 6): # Creating each eigenblock
    eigenblock = pca.components_[i].reshape(32, 32, 3)
    
    min_val = eigenblock.min()
    max_val = eigenblock.max()
    eigenblock_disp = (eigenblock - min_val) / (max_val - min_val)
    
    plt.imshow(eigenblock_disp)
    plt.title(f"Eigenblock {i}")
    plt.axis('off')
    plt.savefig(f"Eigenblock {i}.png")
    print(f"Saved as Eigenblock {i}.png")
    plt.clf()

i = 15 # Just picked a random source photo
reconstructed = pca.inverse_transform(X_pca)

recon_disp = reconstructed.reshape(32, 32, 3)

recon_disp -= recon_disp.min()
recon_disp /= recon_disp.max()

plt.imshow(recon_disp)
plt.title("Reconstructed Image")
plt.savefig(f"Reconstructed block {i}.png")
print(f"Reconstructed block {i}.png")

# print("Explained variance per eigenblock:", pca.explained_variance_)
print("Explained variance ratio per eigenblock:", pca.explained_variance_ratio_)