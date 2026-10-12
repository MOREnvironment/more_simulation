from __future__ import annotations


class ComponentParameters:
    location = [0.8, 0.0, -0.2]
    frame_id = 'dvl_link'
    publish_tf = True
    topic = 'dvl/data'
    rate_hz = 5.0
    noise_enabled = True
    random_seed = 2
    dropout_probability = 0.0
    with_covariance = True
    scale_factor = [1.0, 1.0, 1.0]
    bias = [0.0, 0.0, 0.0]
    white_noise_std_per_sample = [0.02, 0.02, 0.02]
    bias_random_walk_std = [0.0, 0.0, 0.0]
