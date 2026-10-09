# Session 13 preregistered protocol

Research question: at approximately 20M parameters and 50M training targets, does reconstructing reversible hidden states reduce measured peak T4 memory, and does the gain translate into a larger useful physical batch?

The five principal conditions are A conventional stored, B selected reversible reconstructed at a common physical batch, C selected reversible reconstructed at its maximum physical batch, D identical reversible equations with stored autograd, and E selected reversible reconstructed at the common physical batch with accumulation to C's effective batch. Each condition must commit exactly 50,000,000 valid targets. A, B, and C are repeated with a second initialization seed if the live Kaggle quota permits. Every divergence is recorded rather than omitted.

Before full training, test conventional, midpoint h=0.25/0.5/1.0, Euler-style coupling h=0.25/0.5/1.0, and blended midpoint a=0.5 h=0.25 for 2M targets each. Reject numerical failures; select lowest development loss among passing candidates, breaking differences within 0.02 nats by gradient disagreement then throughput. The pilot will not be counted toward any principal run.

Model and measurement choices are locked in `configs/study.json`. Failed attempts, validation, padding, download, compilation, and warm-up tokens are not completed training. Report GPU memory allocated and reserved separately. The final write-up must distinguish architectural effects (A vs D), reconstruction effects (D vs B), effective batch effects (B vs E), and physical batch effects (E vs C).

Primary source: Gal et al., *Reversing Large Language Models for Efficient Training and Fine-Tuning*, arXiv:2512.02056 (2025). Course source: Session 13 GMeet transcript around 90:48-91:53 and the Session 13 lecture. The report will attribute claims to the source and never infer assignment results from literature.
