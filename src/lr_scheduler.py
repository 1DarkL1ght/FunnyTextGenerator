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
        self.lr1 = lr1
        self.lr2 = lr2
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch

        if step < self.num_warmup_steps:
            progress = step / max(1, self.num_warmup_steps)

            lr_mapping = self.lr1 + progress * (self.lr2 - self.lr1)
            lr_backbone = progress * self.lr2
            
            return [lr_mapping, lr_backbone]

        decay_steps = self.num_training_steps - self.num_warmup_steps
        current_decay_step = step - self.num_warmup_steps

        decay_progress = current_decay_step / max(1, decay_steps)
        decay_mult = max(0.0, 1.0 - decay_progress)

        lr_mapping = self.lr2 * decay_mult
        lr_backbone = self.lr2 * decay_mult

        return [lr_mapping, lr_backbone]
