from __future__ import annotations


class ComponentParameters:
    location = [0.0, 0.0, 0.0]
    frame_id = 'gnss_link'
    publish_tf = True
    topic = 'gnss/fix'
    rate_hz = 5.0
    noise_enabled = True
    random_seed = 3
    dropout_probability = 0.0
    fix_location = [45.8007257, 15.9721655, 0.0]
    position_bias = [0.0, 0.0, 0.0]
    position_white_noise_std_per_sample = [0.5, 0.5, 1.0]
    position_bias_random_walk_std = [0.0, 0.0, 0.0]
