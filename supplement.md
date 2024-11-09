| Model     | Metric     | DBLP                | IMDB               | ACM                | DBLP Runtime (ms) |
| --------- | ---------- | ------------------- | ------------------ | ------------------ | ----------------- |
| HiNormer  | Macro-F1   | 0.9386 ± 0.0021     | 0.6469 ± 0.0112    | bad result         | 79.15             |
|           | Micro-F1   | 0.9425 ± 0.0019     | 0.6781 ± 0.0058    | bad result         |                   |
| SlotGAT   | Macro-F1   | 94.2 ± 0.4          | 63.8 ± 0.9         | 93.7 ± 0.4         | 186.6             |
|           | Micro-F1   | 94.6 ± 0.4          | 68.4 ± 0.5         | 93.6 ± 0.4         |                   |
| SeHGNN    | Macro-F1   | 94.30 ± 0.41        | 65.89 ± 0.38       | 93.07 ± 0.44       | 19.2              |
|           | Micro-F1   | 94.72 ± 0.41        | 68.40 ± 0.42       | 92.99 ± 0.45       |                   |
| IHSGNN    | Macro-F1   | 94.35 ± 0.39        | 65.93 ± 0.52       | 93.02 ± 0.39       | 25.08             |
|           | Micro-F1   | 94.65 ± 0.24        | 68.75 ± 0.63       | 92.96 ± 0.42       |                   |
| IHSGNN    | Macro-F1   | 91.77 ± 0.64        | 63.39 ± 0.82       | 91.43 ± 0.49       | 16.81             |
| (no joint)| Micro-F1   | 91.94 ± 0.58        | 66.78 ± 0.89       | 91.37 ± 0.42       |                   |
| IHSGNN    | Macro-F1   | 91.58 ± 0.45        | 64.71 ± 0.85       | 92.23 ± 0.51       | 23.77             |
| (no LSP)  | Micro-F1   | 92.18 ± 0.47        | 67.05 ± 0.90       | 92.15 ± 0.55       |                   |
| IHSGNN    | Macro-F1   | 92.73 ± 0.35        | 64.93 ± 0.65       | 92.78 ± 0.32       | 24.35             |
| (no ST)   | Micro-F1   | 93.19 ± 0.30        | 67.65 ± 0.60       | 92.75 ± 0.35       |                   |

Thank you for reviewing our additional experiments. We reproduced SlotGAT, SeHGNN, and HiNormer using their source code and best set in a local environment. 
The experimental results showed that, aside from SlotGAT demonstrating a clear advantage on the ACM dataset (at a significant time cost), 
IHSGNN maintains a performance advantage on both the DBLP and IMDB datasets. Unfortunately, most results fall within the variance range, 
consistent with the findings of [1]. Therefore, we included the per-epoch training time on DBLP to further validate our contribution.

### Regarding Ablation Experiments:
- **no joint**: Excludes the joint learning phase, using only the GNN backbone (essentially representing GAT & GraphSage in our local experiments).
- **no LSP**: Excludes the LSP module.
- **no ST**: Excludes the semantic transformation layer.

One unexpected outcome was our inability to fully reproduce the GAT performance reported in [2] within our framework (with an F1-score discrepancy of over 2 percentage points). 
This highlights the significant performance boost that our joint learning phase provides to the backbone model, improving the F1-score by around 3 percentage points. 
This improvement largely stems from our multi-task distillation two-tower module： a component that we have confirmed to be very effective in real-world large-scale recommendation systems.

Additionally, our paper's ablation experiments included only the **no LSP & ST** configuration because the ST layer also serves to maintain dimensional consistency：
Assuming a hidden dimension of `d` and an initial label sequence embedding dimension of `x`, we first project the `k`-hop neighborhood to `d-k*x`, 
since the LSP module generates supplementary embedding dimensions of `k*x` for `k` hops. Thus, in the **no LSP** scenario, 
we adjusted the semantic transformation layer to `(input, output) -> (d, d)`. In the **no ST** configuration, we added a subsequent semantic transformation layer to adjust `(input, output) -> (d+k*x, d)`.

The results demonstrate that the LSP module plays a critical role in performance improvement.

If you have any questions about the results of the new baseline, please provide the corresponding issues for further discussion. Thank you.

[1] Zhao, T., Yang, C., "Space4hgnn: A novel, modularized and reproducible platform to evaluate heterogeneous graph neural network." In SIGIR '22, pp. 2776–2789, 2022.

[2] Lv, Q., Ding, M., "Are we really making much progress?: Revisiting, benchmarking and refining heterogeneous graph neural networks." In KDD '21, pp. 1150–1160, 2021.
