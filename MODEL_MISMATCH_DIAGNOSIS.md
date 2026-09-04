TRAIN_MODEL:
`HandsGRU(input_size=126, hidden_size=128, num_layers=2, dropout=0.3, num_classes=60)` defined in `train_60_class.py`.

SERVER_MODEL:
`HandsGRU(input_size=126, hidden_size=128, num_layers=2, dropout=0.3, num_classes=num_classes)` defined in `server.py`.

SAVE_OPERATION:
`torch.save(model.state_dict(), model_path)` in `train_60_class.py` line 229 saves the complete state dict (10 keys).

CURRENT_CHECKPOINT:
32.7 KB (32,757 bytes)
['fc.weight', 'fc.bias']

OVERWRITE_FOUND:
NO
There is no code in the repo overwriting `gru_model.pt`. The file timestamp is 02-09-2026 01:31:24 (from before current execution session).

TRAINING_VS_CHECKPOINT_DISCREPANCY:
The previous training run in task-146 aborted with `ValueError: Number of classes, 44, does not match size of target_names, 60` during evaluation on line 220, BEFORE reaching line 229 (`torch.save`). Thus, `gru_model.pt` was NEVER updated by `train_60_class.py` and remained the old truncated 32.7 KB checkpoint.

INPUT_SHAPE_ISSUE:
Raw `.npz` files store un-flattened landmark arrays of shape `(36, 2, 21, 3)`. In `train_60_class.py`, `LandmarkDataset` calls `feats.reshape(36, 126)` to flatten `2 * 21 * 3 = 126` floats per frame before feeding into the GRU.

ROOT_CAUSE:
`train_60_class.py` crashed at `classification_report()` (line 220) due to unrepresented classes in the test split before `torch.save()` (line 229) could run. Consequently, `gru_model.pt` on disk was never replaced and remains an incomplete 32.7 KB file.

SAFE_NEXT_ACTION:
Fix the `zero_division`/`labels` handling in line 220 of `train_60_class.py` so training completes through line 229, properly saving the complete 797.7 KB 60-class GRU model checkpoint.
