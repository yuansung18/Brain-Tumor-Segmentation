from functools import partial

from .misc import (
    ce_minus_log_dice,
    weighted_cross_entropy,
    minus_dice,
    l2_plus_kl,
)


loss_function_hub = {
    'crossentropy-log[my_dice]': partial(ce_minus_log_dice, dice_type='my'),
    'crossentropy-log[generalized_dice]': partial(ce_minus_log_dice, dice_type='generalized'),
    'crossentropy-log[naive_dice]': partial(ce_minus_log_dice, dice_type='naive'),
    'my_dice': partial(minus_dice, dice_type='my'),
    'generalized_dice': partial(minus_dice, dice_type='generalized'),
    'naive_dice': partial(minus_dice, dice_type='naive'),
    'sigmoid_dice': partial(minus_dice, dice_type='sigmoid'),
    'crossentropy': weighted_cross_entropy,
    'l2+kl': partial(l2_plus_kl, weight_L2=0.1, weight_KL=0.1),
}
