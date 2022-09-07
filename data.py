import scipy.io as sio

def load_data():
    bad_ROIs = sio.loadmat('Data/bad_ROIs.mat')['bad_ROIs']
    df_F = sio.loadmat('Data/df_F.mat')['df_F']
    events_above_min = sio.loadmat('Data/events_above_min.mat')['events_above_min']
    include_frames = sio.loadmat('Data/include_frames.mat')['include_frames']
    quad_data_norm = sio.loadmat('Data/quad_data_norm.mat')['quad_data_norm']
    scaled_centroids = sio.loadmat('Data/scaled_centroids.mat')['scaled_centroids']

    df_F = df_F[include_frames[0][0] - 1:include_frames[1][0] - 1, :]
    df_F = np.delete(df_F, np.squeeze(bad_ROIs) - 1, axis=1)
    events_above_min = events_above_min[include_frames[0][0] - 1:include_frames[1][0] - 1, :]
    events_above_min = np.delete(events_above_min, np.squeeze(bad_ROIs) - 1, axis=1)
    scaled_centroids = np.delete(scaled_centroids, np.squeeze(bad_ROIs) - 1, axis=0)
    quad_data_norm = np.squeeze(quad_data_norm)[include_frames[0][0] - 1:include_frames[1][0] - 1]

    return bad_ROIs, df_F, events_above_min, include_frames, quad_data_norm, scaled_centroids