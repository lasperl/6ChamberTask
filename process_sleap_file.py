import yaml
import os
import numpy as np

from sleap_nn.predict import run_inference

# FUNCTIONS
def export_dense_predictions(labels, out_file):
    video = labels.videos[0]
    n_frames = len(video)

    skeleton = labels.skeletons[0]
    node_names = [node.name for node in skeleton.nodes]
    n_nodes = len(node_names)

    # coordinates
    xy = np.full((n_frames, n_nodes, 2), np.nan, dtype=np.float32)

    # optional confidence scores
    scores = np.full((n_frames, n_nodes), np.nan, dtype=np.float32)

    for lf in labels.labeled_frames:

        if len(lf.predicted_instances) == 0:
            continue

        inst = lf.predicted_instances[0]

        # PredictedPointsArray
        xy[lf.frame_idx] = inst.points["xy"]

        if "score" in inst.points.dtype.names:
            scores[lf.frame_idx] = inst.points["score"]

    np.savez_compressed(
        out_file,
        xy=xy,
        scores=scores,
        node_names=np.array(node_names, dtype=object),
        n_frames=n_frames,
    )

    #return xy, scores


# info
folder = "/data07/Lina/6Chamber_LbO/Data_analysis"
folder_raw = '/data07/Lina/6Chamber_LbO/Data_collection'

model_dir = "/data07/Lina/6Chamber_LbO/Data_analysis/SLEAP/SingleAnimal/test_model/260828_152828.single_instance.n=370/" # Folder containing SLEAP model

rats = [
    # 'F_B1C1R1',
    # 'F_B1C1R3',
    # 'M_B1C1R1',
    # 'M_B1C1R3',
    # 'F_B2C1R1',
    # 'F_B2C1R3',
    #'M_B2C1R1',
    # 'M_B2C1R3',
    # 'F_B3C1R1',
    # 'F_B3C1R3',
    # 'M_B3C1R1',
    'M_B3C1R3'
]

phases = [
    'Baseline_Closed',
    # 'Baseline_Open',
    # 'Observation_Neutral',
    # 'Observation_Shock',
    # 'Recall_Closed',
    # 'Recall_Open'
]

# MAIN SCRIPT
# Load common parameters
#overall_info = yaml.safe_load(open(os.path.join(folder, 'parameters.yaml')))

for rat in rats:
    comp_phases = phases

    # load time sync data
    rat_folder = os.path.join(folder, '{}/Behavior'.format(rat))
    # sync = h5py.File(os.path.join(rat_folder, 'sync.hdf5'),'r')

    # load rat meta data
    rat_info = yaml.safe_load(open(os.path.join(folder, rat, 'meta_data.yaml')))

    for phase in comp_phases:

        print('processing: {}, {}'.format(rat, phase))
        folder_analysis = os.path.join(rat_folder, phase)

        data = {}

        repo = os.path.join(rat_folder, phase)

        repo_raw = os.path.join(folder_raw, '{}/Behavior/{}').format(rat, phase)

        video = [s for s in os.listdir(repo_raw) if s.endswith('.avi')][-1]
        video_path = os.path.join(repo_raw, video)

        # Run inference and save predictions
        labels = run_inference(
            data_path=video_path,
            model_paths=[model_dir],
            output_path=os.path.join(repo, 'sleap_data', 'SLEAP_predictions.slp'),  # where predictions will be saved
            device="cuda"  # or "cpu"
        )

        export_dense_predictions(labels, os.path.join(repo, 'sleap_data', 'missinginstanceSLEAP_predictions_eye.npz'))
