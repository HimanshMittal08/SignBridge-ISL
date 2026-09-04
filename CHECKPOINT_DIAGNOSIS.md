CHECKPOINTS_FOUND:
- artifacts/isl-inference/models/gru_model.pt | 32.7 KB (32,757 bytes) | PARTIAL
- artifacts/isl-inference/models/backup_8class/gru_model.pt | 797.7 KB (797,733 bytes) | FULL_GRU

CURRENT_SAVE_LOGIC:
`train_60_class.py` saves `model.state_dict()` via `torch.save(model.state_dict(), model_path)` at line 229 after training.

CURRENT_LOAD_EXPECTATION:
`server.py` expects a complete `HandsGRU` state dict with GRU weights (`gru.*`) and linear classifier weights (`fc.*`) matching `num_classes = len(LABEL_MAP)` (60 classes).

ROOT_CAUSE:
The 60-class checkpoint `artifacts/isl-inference/models/gru_model.pt` contains only linear layer weights (`['fc.weight', 'fc.bias']`) and is missing all GRU layer parameters (`gru.weight_ih_l0`, etc.), causing PyTorch `load_state_dict` to fail when initializing `HandsGRU(num_classes=60)`.

RECOVERABLE_WITH_EXISTING_CHECKPOINT: NO
BEST_EXISTING_CHECKPOINT: NONE (artifacts/isl-inference/models/backup_8class/gru_model.pt is a full GRU model but only has 8 classes, not 60)

SAFE_NEXT_ACTION:
Run `python train_60_class.py` using the project virtual environment (`v/Scripts/python.exe`) to train and generate a full 60-class GRU model checkpoint containing all GRU and FC layer weights.
