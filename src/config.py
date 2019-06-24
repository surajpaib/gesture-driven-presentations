# Simple configuration file for most of the constants that we use in the project.
# Not really worth going with something much more complicated (i.e. a config package) at this moment.
#
# I tried to group them up somewhat logically, but it's probably not perfect.

CONFIG = {
    "model_folder": "..\\openpose\\models",
    "model_pose": "COCO",
    "net_resolution": "176x-1",

    "interpolation_frames": 8,
    "matrix_size": 32,
    "used_keypoints": ["RWrist", "LWrist"],
    "confidence_threshold": 0.3,
    "matrix_vertical_crop": 10,

    "presentation_path": "../MRP-6.pptx",

    "correlation_classifier_dataset": "dataset\\",

    "use_dilation": False,
    "kernel_size": 2,

    "reconstruction_error_threshold": 40,

    "xml_files_path": "xml_files",

    # process_videos.py
    "video_dir": "F:\\MRP6 data\\reset_start_stop",
    "result_dir": "Processing_Results_Reset_Start_Stop",
    "result_dir_7": "Processing_Results_Reset_Start_Stop_7",
    "label": "Start_Stop",
    
    # autoencoder.py
    "autoencoder_img_path": ".\\autoencoder_img",
    "latent_space_dim": 100,
    "batch_size": 32,
    "num_epochs": 200,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "noise_frames": 2,
}