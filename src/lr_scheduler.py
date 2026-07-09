from torch.optim.lr_scheduler import _LRScheduler


# def custom_lr_scheduler(optimizer, num_warmup_steps, num_training_steps, lr1, lr2):
#     def lr_lambda_group_0(current_step):
#         """Нестандартный: от lr1 до lr2 за время warmup, затем спад как у первого"""
#         if current_step < num_warmup_steps:
#             # Считаем прогресс warmup от 0 до 1
#             progress = float(current_step) / float(max(1, num_warmup_steps))
#             # Стартуем с множителя, который даст lr1, и идем к 1.0 (который даст lr2)
#             # Формула: start_multiplier + progress * (1 - start_multiplier)
#             # Где start_multiplier = lr1 / lr2
#             start_mult = lr1 / lr2
#             return start_mult + progress * (1.0 - start_mult)
        
#         # После warmup повторяет логику первого (спад от 1.0 до 0)
#         return lr_lambda_group_1(current_step)

#     def lr_lambda_group_1(current_step):
#         """Стандартный: warmup от 0 до 1, затем линейный спад до 0"""
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         return max(
#             0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
#         )
    
#     return LambdaLR(optimizer, lr_lambda=[lr_lambda_group_0, lr_lambda_group_1])

class CustomWarmupDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, num_training_steps, lr1, lr2, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.start_mult = lr1 / lr2
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.num_warmup_steps:
            progress = step / max(1, self.num_warmup_steps)
            mult_0 = self.start_mult + progress * (1.0 - self.start_mult)
            mult_1 = progress
            return [base_lr * mult_0 if i == 0 else base_lr * mult_1 for i, base_lr in enumerate(self.base_lrs)]
        
        decay_progress = (self.num_training_steps - step) / max(1, self.num_training_steps - self.num_warmup_steps)
        decay_mult = max(0.0, decay_progress)
        return [base_lr * decay_mult for base_lr in self.base_lrs]