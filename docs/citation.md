# Citation

If INFTY or its included algorithms are useful in your work, please cite the corresponding software and method papers.

## Software citation

For the toolkit, use a software citation similar to:

```bibtex
@software{infty_toolkit,
  title        = {INFTY: An Optimization Toolkit to Support Continual AI},
  author       = {INFTY contributors},
  year         = {2026},
  url          = {https://github.com/THUDM/INFTY}
}
```

Replace the year, version, DOI, or official repository URL if a formal release is created.

## Algorithm papers

INFTY currently exposes three optimizer families under `infty.optim`. The papers below are grouped by family, and each family is ordered by year from newest to oldest.

### Geometry reshaping

<table class="algo-papers">
  <thead>
    <tr>
      <th>Method</th>
      <th>Venue</th>
      <th>Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>C-Flat++</code></td>
      <td>arXiv 2025</td>
      <td><a href="https://arxiv.org/abs/2508.18860"><em>C-Flat++: Towards a More Efficient and Powerful Framework for Continual Learning.</em></a></td>
    </tr>
    <tr>
      <td><code>C-Flat</code></td>
      <td>arXiv 2024</td>
      <td><a href="https://arxiv.org/abs/2404.00986v2"><em>Make Continual Learning Stronger via C-Flat.</em></a></td>
    </tr>
    <tr>
      <td><code>GAM</code></td>
      <td>CVPR 2023</td>
      <td><a href="https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_Gradient_Norm_Aware_Minimization_Seeks_First-Order_Flatness_and_Improves_Generalization_CVPR_2023_paper.html"><em>Gradient Norm Aware Minimization Seeks First-Order Flatness and Improves Generalization.</em></a></td>
    </tr>
    <tr>
      <td><code>LookSAM</code></td>
      <td>CVPR 2022</td>
      <td><a href="https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Towards_Efficient_and_Scalable_Sharpness-Aware_Minimization_CVPR_2022_paper.html"><em>Towards Efficient and Scalable Sharpness-Aware Minimization.</em></a></td>
    </tr>
    <tr>
      <td><code>GSAM</code></td>
      <td>ICLR 2022</td>
      <td><a href="https://openreview.net/forum?id=edONMAnhLu-"><em>Surrogate Gap Minimization Improves Sharpness-Aware Training.</em></a></td>
    </tr>
    <tr>
      <td><code>SAM</code></td>
      <td>ICLR 2021</td>
      <td><a href="https://openreview.net/forum?id=6Tm1mposlrM"><em>Sharpness-Aware Minimization for Efficiently Improving Generalization.</em></a></td>
    </tr>
  </tbody>
</table>

Implementation mapping: `C_Flat` exposes both `C-Flat` with `strategy="basic"` and `C-Flat++` with `strategy="plus"`.

### Zeroth-order updates

<table class="algo-papers">
  <thead>
    <tr>
      <th>Method</th>
      <th>Venue</th>
      <th>Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>ZeroFlow</code></td>
      <td>arXiv 2025</td>
      <td><a href="https://arxiv.org/abs/2501.01045"><em>Zeroflow: Overcoming Catastrophic Forgetting Is Easier Than You Think.</em></a></td>
    </tr>
    <tr>
      <td><code>ZeroFlow (MeZO-style variants)</code></td>
      <td>NeurIPS 2023</td>
      <td><a href="https://arxiv.org/abs/2305.17333"><em>Fine-Tuning Language Models with Just Forward Passes.</em></a></td>
    </tr>
    <tr>
      <td><code>ZeroFlow (forward gradient)</code></td>
      <td>arXiv 2022</td>
      <td><a href="https://arxiv.org/abs/2202.08587"><em>Gradients without Backpropagation.</em></a></td>
    </tr>
  </tbody>
</table>

Implementation mapping: `ZeroFlow` uses `inftyopt="forward_grad"` for the forward-gradient path, and uses `zo_sgd`, `zo_adam`, `zo_sgd_sign`, `zo_adam_sign`, `zo_sgd_conserve`, or `zo_adam_conserve` for the MeZO-style zeroth-order paths.

### Gradient filtering

<table class="algo-papers">
  <thead>
    <tr>
      <th>Method</th>
      <th>Venue</th>
      <th>Paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>UniGrad-FS</code></td>
      <td>IEEE TII 2024</td>
      <td><a href="https://ieeexplore.ieee.org/abstract/document/10636267"><em>UniGrad-FS: Unified Gradient Projection With Flatter Sharpness for Continual Learning.</em></a></td>
    </tr>
    <tr>
      <td><code>CAGrad</code></td>
      <td>NeurIPS 2021</td>
      <td><a href="https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html"><em>Conflict-Averse Gradient Descent for Multi-Task Learning.</em></a></td>
    </tr>
    <tr>
      <td><code>GradVac</code></td>
      <td>ICLR 2021</td>
      <td><a href="https://openreview.net/forum?id=F1vEjWK-lH_"><em>Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models.</em></a></td>
    </tr>
    <tr>
      <td><code>PCGrad</code></td>
      <td>NeurIPS 2020</td>
      <td><a href="https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html"><em>Gradient Surgery for Multi-Task Learning.</em></a></td>
    </tr>
    <tr>
      <td><code>OGD</code></td>
      <td>AISTATS 2020</td>
      <td><a href="https://arxiv.org/abs/1910.07104"><em>Orthogonal Gradient Descent for Continual Learning.</em></a></td>
    </tr>
  </tbody>
</table>

For publications, cite the specific paper corresponding to the optimizer class or execution path used in your experiments.

## Reproducibility note

When citing INFTY in experiments, report:

- INFTY version or commit hash;
- optimizer name;
- optimizer `args` dictionary;
- base optimizer and hyperparameters;
- PyTorch version;
- benchmark and task order.
