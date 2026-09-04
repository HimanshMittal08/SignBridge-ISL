MODEL_LOAD: FAIL
LABEL_MAP: PASS
CLASS_COUNT: 60
PREDICT_ENDPOINT: FAIL
VALID_PREDICTION_RETURNED: NO
PREDICTED_LABEL: N/A
CONFIDENCE: N/A
ERROR: RuntimeError: Error(s) in loading state_dict for HandsGRU: Missing key(s) in state_dict: "gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0", "gru.weight_ih_l1", "gru.weight_hh_l1", "gru.bias_ih_l1", "gru.bias_hh_l1" (checkpoint contains only ['fc.weight', 'fc.bias'])

FINAL_VERDICT: FAIL
