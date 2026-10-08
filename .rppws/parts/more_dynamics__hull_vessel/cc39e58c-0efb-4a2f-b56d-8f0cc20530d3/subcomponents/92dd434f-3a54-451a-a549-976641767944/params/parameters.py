from __future__ import annotations


class ComponentParameters:
    location = [0.0, 0.0, 0.0]
    frame_id = "{'name': 'frame_id', 'default_value': '', 'type': 'string'}"
    publish_tf = False
    topic = 'position'
    rate_hz = 1.0
    noise_enabled = True
    random_seed = 3
    dropout_probability = 0.0
    with_covariance = True
    position_scale_factor = [1.0, 1.0, 1.0]
    position_bias = [0.0, 0.0, 0.0]
    position_white_noise_std_per_sample = [0.5, 0.5, 0.5]
    position_bias_random_walk_std = [0.0, 0.0, 0.0]
    orientation_scale_factor = [1.0, 1.0, 1.0]
    orientation_bias = [0.0, 0.0, 0.0]
    orientation_white_noise_std_per_sample = [0.0, 0.0, 0.0]
    orientation_bias_random_walk_std = [0.0, 0.0, 0.0]
