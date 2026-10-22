# U250 strict-contract QAT（远端隔离版本）

## 目的与隔离边界

该版本用于在远端 GPU 主机上模拟当前 U250 strict K24 部署协议，验证
QAT 是否能降低板端 INT8 recurrent-state 误差。它位于：

```
/workspace/SPN_Quantization/.worktrees/cspn-u250-strict-contract-qat
```

已有的 `/workspace/SPN_Quantization/.worktrees/cspn-mixed-task-aware-qat`
及其 `profile_logs/nyu_cspn_mixed_task_aware_qat/` 没有被修改。新实验输出
只写入 `profile_logs/nyu_cspn_u250_strict_contract_qat/`。

## 合同模型

`scripts/u250_strict_contract_qat.py` 的 `U250StrictPropagation` 显式模拟：

- 8 邻域 CSPN、K=24；
- signed INT8 fake-quant + STE，量化范围 [-127, 127]；
- normalized gate scale `1/127`；
- raw/current/output depth scale `14/127 m`；
- 每个 recurrent stage 的 neighbor pack、MUL、sum、residual Add 后重新 QDQ；
- sparse depth 采用硬约束，已知点每轮恢复为 raw depth；
- 不改 RTL、ISA bitstream、DS compiler，也不执行 DMA/U250 写入。

因此它不是通用 QAT 的 `A8/Q13` propagation adapter；两者的数值合同和
训练目标不同。

## 数据与形状

实验使用远端已有的 128 个 NYU calibration HDF5 和 32 个 representative
validation HDF5（不下载、不覆盖）：

```
/workspace/agentflow_sdformer_hw128/data/train_data/
/workspace/agentflow_sdformer_hw128/data/val32/
```

模型采用工作树中已修正的 ResNet18/tiny decoder，输入和官方模型一致为
`[N,4,228,304]`。这验证了 strict propagation 的量化趋势；它与当前 DS
整图的 256×256 前端几何仍有差异，不能直接当成 256 板端最终 RMSE。

## 运行结果

启动 checkpoint：

```
/workspace/CSPN/cspn_pytorch/output/nyu_converged_baselines/cspn_iter24/best.pt
```

### baseline（无 QAT）

32 个 validation 样本、2,217,984 个有效深度像素：

| 配置 | RMSE (m) | MAE (m) |
|---|---:|---:|
| 原始 checkpoint + strict U250 fake-quant | 0.48457 | 0.38105 |

### strict-contract QAT（128 train / 32 val）

| epoch | train L1 (m) | val RMSE (m) | val MAE (m) |
|---:|---:|---:|---:|
| 1 | 0.22208 | 0.26735 | 0.16797 |
| 2 | 0.17913 | 0.26981 | 0.17235 |
| 3 | 0.15913 | 0.25174 | 0.15818 |
| 4 | 0.15090 | 0.24590 | 0.15468 |
| 5 | 0.14819 | **0.24220** | **0.15127** |

输出：

```
/workspace/SPN_Quantization/profile_logs/nyu_cspn_u250_strict_contract_qat/qat_e5/report.json
/workspace/SPN_Quantization/profile_logs/nyu_cspn_u250_strict_contract_qat/qat_e5/strict_contract_qat.pt
```

strict QAT 在同一套远端协议和验证集上将 RMSE 从约 0.485 m 降至约
0.242 m，说明“按 U250 recurrent INT8 合同训练”确实能够显著降低量化误差。
这不是 U250 实板结果，也不证明 256×256 DS 整图已经部署；下一步应将
QAT 得到的权重接入现有 DS strict 256×256 graph，再用 CModel raw 输出和
U250 raw 输出逐元素比较。

