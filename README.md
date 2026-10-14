# CS514: AI Cluster Networking

**Designing AI Training Clusters with Simulation: From GPT-3 to Mixture-of-Experts Workloads**

Fall 2026 course project by **Yiheng Fan** and **Ying Xiong**.

## Project scope

Reproduce the SimAI paper's GPT-3 175B bandwidth and tensor-parallelism case studies (Figures 10 and 11(b)), then investigate a DeepSeek MoE training workload using AICB and SimAI-Analytical.

This repository currently contains the project proposal. No simulation results are claimed at this stage.

## Proposal

- `proposal/paper.tex`: latest English proposal, including the AI-use disclosure.
- `proposal/bibs.bib`: corresponding BibTeX records.

The LaTeX source contains inline references and can compile as a standalone document. The `.bib` file is provided for future bibliography management.

With a LaTeX installation containing the required packages:

```sh
cd proposal
pdflatex paper.tex
pdflatex paper.tex
```

The second pass resolves cross-references. Alternatively, upload `paper.tex` to a LaTeX editor with the required packages and compile it there.

## Reproducibility

As experiments are implemented, this repository will preserve configuration files, scripts, raw outputs, plotting code, and step-by-step rerun instructions. Both team members will contribute through Git commits throughout the project.

## References

- [SimAI paper (NSDI 2025)](https://www.usenix.org/conference/nsdi25/presentation/wang-xizheng-simai)
- [SimAI](https://github.com/aliyun/SimAI)
- [AICB](https://github.com/aliyun/aicb)
