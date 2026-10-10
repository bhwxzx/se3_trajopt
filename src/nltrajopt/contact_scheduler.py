import math


class ContactScheduler:
    def __init__(self, model, dt=0.01, contact_frame_dict=None):

        self.model = model
        self.DT = dt
        self.total_nodes = 0
        self.contact_phases = []
        self.contact_sequence_fnames = []

        self.contact_frame_dict = contact_frame_dict

    def add_phase(self, phase, duration_sec):
        # Floating-point division can put an exact step count just below an integer.
        steps = int(round(duration_sec / self.DT))
        if duration_sec < 0 or not math.isclose(
            steps * self.DT, duration_sec, rel_tol=1e-9, abs_tol=1e-12
        ):
            raise ValueError(
                f"Phase duration {duration_sec} must be a non-negative integer "
                f"multiple of dt={self.DT}"
            )
        self.total_nodes += steps
        self.contact_phases.extend([phase] * steps)

        for _ in range(steps):
            frame_names = []
            for key in phase:
                frame_names.extend(self.contact_frame_dict.get(key, []))

            self.contact_sequence_fnames.append(frame_names)
