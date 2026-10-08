from __future__ import annotations


class ComponentParameters:
    location = [0.2, 0.0, 0.1]
    frame_id = 'imu_link'
    publish_tf = True
    topic = 'imu/data'
    rate_hz = 0.0
    noise_enabled = True
    random_seed = 1
    dropout_probability = 0.0
    gravity = 9.80665
    orientation_bias = [0.0, 0.0, 0.0]
    orientation_white_noise_std_per_sample = [0.01, 0.01, 0.01]
    orientation_bias_random_walk_std = [0.0, 0.0, 0.0]
    gyro_scale_factor = [1.0, 1.0, 1.0]
    gyro_bias = [0.0, 0.0, 0.0]
    gyro_white_noise_std_per_sample = [0.002, 0.002, 0.002]
    gyro_bias_random_walk_std = [0.0, 0.0, 0.0]
    accel_scale_factor = [1.0, 1.0, 1.0]
    accel_bias = [0.0, 0.0, 0.0]
    accel_white_noise_std_per_sample = [0.05, 0.05, 0.05]
    accel_bias_random_walk_std = [0.0, 0.0, 0.0]
