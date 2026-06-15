import numpy as np

from fedrel_journal.methods.hash_teacher import (
    apply_reliability_postprocess,
    fit_reliability_postprocess,
    postprocess_reliability,
)


def test_reliability_postprocess_applies_training_parameters_to_eval_scores():
    train_scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    eval_scores = np.array([3.0, 4.0, 5.0], dtype=np.float32)

    params = fit_reliability_postprocess(train_scores, clip_quantiles=(0.0, 1.0))
    eval_from_train_params = apply_reliability_postprocess(eval_scores, params)
    eval_refit_on_eval = postprocess_reliability(eval_scores, clip_quantiles=(0.0, 1.0))

    assert np.allclose(eval_from_train_params, np.array([1.0, 1.0, 1.0], dtype=np.float32))
    assert not np.allclose(eval_from_train_params, eval_refit_on_eval)
