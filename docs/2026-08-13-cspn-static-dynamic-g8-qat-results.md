# CSPN Static and Dynamic Group-8 W4A4 QAT Results

## Contract

The experiment uses the official CSPN ResNet-18 model with 24 propagation
steps, the real NYU train/validation splits, the converged official checkpoint,
128 stratified train calibration samples, and 64 fixed paired validation
samples. Weights use signed symmetric W4 per output channel. Ordinary
activations use contiguous Group-8 A4, with unsigned codes after ReLU and
signed symmetric codes elsewhere. Guidance and bias remain FP32. Propagation
uses A8 affinity before normalization, signed INT16 Q13 coefficients, INT32
accumulation, A8 state, and exact sparse-depth anchors.

Static QAT retains the activation ranges frozen before fine-tuning. Dynamic
QAT computes ordinary activation scales online per sample. Structural decoder
ranges and propagation ranges remain frozen in both modes. Final evaluation
loads each canonical checkpoint into a fresh official model and uses only the
hard deployment path.

## Convergence

The original `1e-3` learning-rate pilot was unstable even after finite-gradient
checks and global gradient clipping. Both formal runs therefore restarted from
the same official checkpoint at `1e-4`, with SGD momentum `0.9`, weight decay
`1e-4`, global gradient norm limit `10`, and identical data order.

| Mode | Best epoch | Best validation RMSE | Final epoch | Stop reason |
| --- | ---: | ---: | ---: | --- |
| Static-G8 QAT | 9 | 0.231229 | 15 | validation plateau |
| Dynamic-G8 QAT | 5 | 0.221866 | 11 | validation plateau |

Static training diverged after its best checkpoint and reached RMSE `10.3357`
at epoch 15. Evaluation uses the preserved epoch-9 `best.pt`, not the final
weights. Dynamic training also selected its earlier best checkpoint.

## Full-Validation Hard-Path Accuracy

All rows cover the same 654 official CSPN validation samples.

| Configuration | RMSE | MAE | AbsRel | iRMSE | Flat RMSE | Boundary RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 0.148248 | 0.060357 | 0.021260 | 0.024020 | 0.103742 | 0.391914 |
| PTQ Static-G8 W4A4 | 0.339795 | 0.258379 | 0.118009 | 89.890225 | 0.318235 | 0.491301 |
| PTQ Dynamic-G8 W4A4 | 0.312888 | 0.242017 | 0.102403 | 0.090726 | 0.288913 | 0.481672 |
| QAT Static-G8 W4A4 | 0.231229 | 0.158849 | 0.065237 | 11.667228 | 0.202235 | 0.426906 |
| QAT Dynamic-G8 W4A4 | **0.221866** | **0.149574** | **0.059757** | 43.111153 | **0.192844** | 0.429273 |

Static QAT reduces RMSE by `0.108566` (`31.95%`) relative to Static PTQ.
Dynamic QAT reduces RMSE by `0.091022` (`29.09%`) relative to Dynamic PTQ and
is the best tested W4A4 configuration. It remains `49.7%` above FP32 RMSE.

iRMSE is not recovered. A very small number of exact zero predictions dominate
the reciprocal-depth metric: the Dynamic QAT fixed-64 set contains one zero
pixel out of approximately 4.44 million predictions. This is a real endpoint
stability failure even though RMSE, MAE, and AbsRel improve substantially.

## Paired 64-Sample Comparison

| Configuration | Mean RMSE | Mean MAE | Mean AbsRel | Mean iRMSE |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 0.159792 | 0.064772 | 0.021648 | 0.022580 |
| PTQ Static-G8 W4A4 | 0.339473 | 0.255983 | 0.111955 | 59.426573 |
| PTQ Dynamic-G8 W4A4 | 0.323601 | 0.247205 | 0.101105 | 0.085889 |
| QAT Static-G8 W4A4 | 0.237618 | 0.159924 | 0.062386 | 0.047457 |
| QAT Dynamic-G8 W4A4 | **0.234603** | **0.157639** | **0.060330** | 59.398292 |

Static QAT improves all 64 paired samples, with mean per-sample RMSE delta
`-0.101855`. Dynamic QAT improves 58 samples and worsens 6, with mean delta
`-0.088999`.

## Quantization and Propagation Diagnostics

Diagnostics below aggregate the fixed 64-sample activation sites and report
the mean final-step propagation-state MSE.

| Configuration | Activation SQNR (dB) | New-zero rate | Saturation rate | Step-24 state MSE |
| --- | ---: | ---: | ---: | ---: |
| PTQ Static-G8 | 13.017 | 0.3413 | 0.6693 | 0.0007892 |
| PTQ Dynamic-G8 | 15.831 | 0.2333 | 0.6343 | 0.0006924 |
| QAT Static-G8 | **17.168** | **0.2570** | 0.6752 | **0.0005862** |
| QAT Dynamic-G8 | 16.211 | 0.2329 | 0.6540 | 0.0007063 |

Static QAT produces the clearest local recovery: higher aggregate activation
SQNR, fewer new zero codes, and lower final propagation-state error. Dynamic
QAT gains endpoint accuracy without reducing final state MSE relative to its
PTQ baseline, so its benefit comes primarily from task-adapted feature and
depth-head weights rather than uniformly cleaner propagation states.

All four quantized configurations retain zero anchor error, zero Q13
coefficient-sum error, zero contraction violations, INT32 accumulation, and
finite predictions. The large activation saturation statistic includes valid
unsigned zero codes at the lower endpoint and must not be interpreted as
upper-tail clipping alone.

## Prediction Figures

Runtime figures are stored outside Git:

- `/workspace/SPN_Quantization/profile_logs/nyu_cspn_group_a4_qat/figures/paired_prediction_details.png`
- `/workspace/SPN_Quantization/profile_logs/nyu_cspn_group_a4_qat/figures/paired_prediction_details.pdf`
- `/workspace/SPN_Quantization/profile_logs/nyu_cspn_group_a4_qat/figures/paired_predictions_64.png`
- `/workspace/SPN_Quantization/profile_logs/nyu_cspn_group_a4_qat/figures/paired_predictions_64.pdf`

The detailed panel selects the four samples with the largest quantized error
and shows RGB, sparse depth, GT, FP32, both PTQ predictions, both QAT
predictions, and their absolute-error maps.

## Conclusion

Strict W4A4 QAT materially recovers CSPN depth accuracy. Dynamic-G8 QAT gives
the best RMSE, MAE, and AbsRel, while Static-G8 QAT gives the cleaner
propagation and reciprocal-depth behavior on the paired subset. Neither mode
matches FP32, and rare exact-zero endpoint failures make iRMSE unacceptable.
The next targeted change should constrain the final depth endpoint to remain
strictly positive without relaxing the rest of the W4A4 contract.
