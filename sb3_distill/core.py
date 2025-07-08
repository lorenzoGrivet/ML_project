from abc import ABC

class PolicyDistillationAlgorithm(ABC):
    def set_teacher(self, teacher_model, teacher_env=None):
        #set the teacher model and environment for policy distillation
        self.teacher_model = teacher_model
        self.teacher_env = teacher_env

    def _excluded_save_params(self):
        # exclude the teacher model from being saved
        excluded = super()._excluded_save_params()
        return excluded + ['teacher_model']
