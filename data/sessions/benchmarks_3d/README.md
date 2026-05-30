# 3D-Only Benchmark Sessions

These sessions are intended for tuning `--piano-trigger-mode 3d`, where Record3D `height_above_desk_m` is the main hit/release signal and 2D tracking is used mainly to choose the key.

- `bench3d_index_slow_g4`: right index finger slowly taps `G4`; expected to trigger once per deliberate tap.
- `bench3d_index_fast_g4`: right index finger rapidly taps `G4`; expected to trigger each fast tap without requiring exaggerated lift.
- `bench3d_ring_low_lift_e4`: right ring finger naturally taps `E4` with low lift; used to tune low-lift finger sensitivity.
- `bench3d_pinky_low_lift_f4`: right pinky naturally taps `F4` with low lift; used to tune low-lift finger sensitivity.
- `bench3d_rest_on_keys`: fingers rest naturally on the keys without intentional taps; expected to produce no hits.
- `bench3d_hover_no_contact`: finger hovers above a key without touching the desk; expected to produce no hits.
- `bench3d_adjacent_keys_g4_a4`: right index alternates between adjacent `G4` and `A4`; used to test key selection near neighboring keys.
