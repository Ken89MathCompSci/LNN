import scipy.io as sio

data = sio.loadmat('Source Code\\preprocessed_datasets\\ukdale\\ukdale1.mat')
print("Keys:", data.keys())
print("labelOut:", data['labelOut'])
print("labelOut shape:", data['labelOut'].shape)
print("input shape:", data['input'].shape)
print("output shape:", data['output'].shape)
