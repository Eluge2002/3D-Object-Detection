import numpy as np


class KalmanCV3D:
    """
    Constant-velocity Kalman filter for 3D position:
      state x = [cx, cy, z, vx, vy, vz]^T

    Measurement z_meas = [cx, cy, z]^T  (can be partial/missing)
    """

    def __init__(self, dt=1.0):
        self.dt = float(dt)

        # State
        self.x = np.zeros((6, 1), dtype=np.float32)

        # Covariance
        self.P = np.eye(6, dtype=np.float32) * 1000.0

        # Motion model
        self.F = np.eye(6, dtype=np.float32)
        self._update_F()

        # Process noise (tune)
        # We'll use a simple diagonal Q scaled by dt
        self.q_pos = 1.0
        self.q_vel = 10.0
        self.Q = np.eye(6, dtype=np.float32)
        self._update_Q()

        # Measurement model for full [cx,cy,z]
        self.H_full = np.zeros((3, 6), dtype=np.float32)
        self.H_full[0, 0] = 1.0
        self.H_full[1, 1] = 1.0
        self.H_full[2, 2] = 1.0

        # Measurement noise (tune)
        self.r_xy = 25.0   # px variance
        self.r_z = 0.25    # relative depth variance
        self.R_full = np.diag([self.r_xy, self.r_xy, self.r_z]).astype(np.float32)

        self.initialized = False

    def _update_F(self):
        dt = self.dt
        self.F[:] = np.eye(6, dtype=np.float32)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

    def _update_Q(self):
        self.Q = np.diag([self.q_pos, self.q_pos, self.q_pos, self.q_vel, self.q_vel, self.q_vel]).astype(np.float32)

    def set_dt(self, dt):
        self.dt = float(dt)
        self._update_F()
        self._update_Q()

    def set_process_noise(self, q_pos=1.0, q_vel=10.0):
        self.q_pos = float(q_pos)
        self.q_vel = float(q_vel)
        self._update_Q()

    def set_measurement_noise(self, r_xy=25.0, r_z=0.25):
        self.r_xy = float(r_xy)
        self.r_z = float(r_z)
        self.R_full = np.diag([self.r_xy, self.r_xy, self.r_z]).astype(np.float32)

    def init_state(self, cx, cy, z, vx=0.0, vy=0.0, vz=0.0):
        self.x = np.array([[cx], [cy], [z], [vx], [vy], [vz]], dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 100.0
        self.initialized = True

    def predict(self):
        if not self.initialized:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update_full(self, cx, cy, z):
        """
        Update with full measurement [cx,cy,z].
        """
        if not self.initialized:
            self.init_state(cx, cy, z)
            return self.x.copy()

        z_meas = np.array([[cx], [cy], [z]], dtype=np.float32)
        H = self.H_full
        R = self.R_full

        y = z_meas - (H @ self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

        return self.x.copy()

    def update_xy_only(self, cx, cy):
        """
        Update with partial measurement [cx,cy] (no depth).
        """
        if not self.initialized:
            # if we don't have z, initialize with z=0
            self.init_state(cx, cy, 0.0)
            return self.x.copy()

        z_meas = np.array([[cx], [cy]], dtype=np.float32)

        H = np.zeros((2, 6), dtype=np.float32)
        H[0, 0] = 1.0
        H[1, 1] = 1.0

        R = np.diag([self.r_xy, self.r_xy]).astype(np.float32)

        y = z_meas - (H @ self.x)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ H) @ self.P

        return self.x.copy()

    def get_state(self):
        if not self.initialized:
            return None
        return self.x.flatten().tolist()
