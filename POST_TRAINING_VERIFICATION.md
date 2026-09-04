MODEL_LOAD: FAIL
STATE_DICT_MATCH: FAIL
LABEL_MAP: PASS
CLASS_COUNT: 60
TEST_INPUT_SHAPE: (36, 2, 21, 3)
INFERENCE: FAIL
PREDICTED_LABEL: N/A
CONFIDENCE: N/A
LABEL_VALID: NO
ERROR: RuntimeError: Missing key(s) in state_dict: "gru.weight_ih_l0", "gru.weight_hh_l0", "gru.bias_ih_l0", "gru.bias_hh_l0", "gru.weight_ih_l1", "gru.weight_hh_l1", "gru.bias_ih_l1", "gru.bias_hh_l1" (checkpoint contains only ['fc.weight', 'fc.bias'])

FINAL_VERDICT: FAIL
