# We still don't know if the models want to learn

Across the One Layer Deeper challenge, we received **15,602 accepted uploads from 206 GitHub accounts**.

## The challenge

The public problem is repeated modular squaring:

```python
z = x % N
for _ in range(T):
    z = z * z % N
```

Our [launch post](https://blog.tilderesearch.com/blog/one-layer-deeper) asked participants to design architectures, optimizers, and losses that learn this computation and use greater depth at test time. Most machine learning methodologies were allowed, but using an arithmetic solver or generating additional training targets was prohibited.

Hard allowed one H100, one hour of training, 30 minutes of evaluation, and 500 million trainable parameters. Certification required every example to be correct at consecutive rungs: T=1, 2, 4, 8, 16, 32, 64.

Only **18 Hard uploads from five accounts** certified any depth on seen-modulus tests: 16 reached T=64, one T=32, and one T=8. Two accounts had rejected entries as they extracted the training data from the sandbox prior to this, violating the metric-recorder rule. The other 143 ranked accounts failed to be certified at T=1.

Thanks to az and apaz for submissions that helped us refine the task and rules during the beta period.

## What the three leaders did

**Hard reversed the decimal digits of N in the prompt.** If N was `253`, it appeared in the Hard dataset as `352`; x, T, and the answer kept their normal digit order. The recurrence was still modular squaring.

We discuss each leader's latest archived Hard upload with 100% aggregate accuracy: CodeReclaimers' `58311e4a`, Yash Kant's `86e9acf2`, and Sahil Verma's `9fa00118`. All certified T=64 on both seen and unseen moduli.

![Three illustrated mechanisms: computed residue transitions, learned digit cells, and selection among supplied programs.](assets/submission-ideas.svg)

**CodeReclaimers (`58311e4a`) selects an arithmetic law, then computes its remainder tables.** It compares candidate laws with learned predictions. After selecting a law, it constructs the transitions directly with this exact expression:

```python
(r + D * r * (r - 1) // 2) % m
```

For D=2, this is `r*r % m`. The code caches maps for 1, 2, 4, and more transitions, then composes them using the bits of T. Separate small-modulus results constrain the final integer answer.

**Yash Kant (`86e9acf2`) learned digit operations inside a supplied calculator.** Small networks predict multiplication, addition, subtraction, carries, and borrows. A multiplication cell can learn `7 × 8 + carry 3 = 59`: output digit 9 and carry 5.

The supplied procedure repeats those learned operations. In pseudocode:

```python
for _ in range(T):
    squared = square_using_learned_digit_cells(z)
    z = remainder_using_learned_digit_cells(squared, N)
```

The digit cells are trained from arithmetic targets. The complete answer circuit is not trained end to end.

**Sahil Verma (`9fa00118`) selected from supplied exact programs.** The code contains 63,525 polynomial formulas, including squaring. Four input-reading choices produce 254,100 candidates. Examples eliminate incorrect candidates; learned weights select the formula and output placement.

```python
candidates = keep_programs_matching(supplied_programs, training_examples)
chosen = learn_which_program_to_select(candidates)
answer = execute_exactly(chosen, x, N, T)
```

## The winner

The [written rules](https://github.com/tilde-research/one-layer-deeper#rules) have restrictions on learning, resources, and data. None of the current top submissions fully comply with the rules.

| Rule | Effect on the learning claim |
|---|---|
| **7:** “No hard-coded algorithm in the forward pass.” | Sahil selects from exact programs. CodeReclaimers directly computes selected arithmetic laws. |
| **8:** “End-to-end learning only.” | Yash trains components separately from the full answer circuit. |

All three leaderboard finalists and Apaz deserve huge credit for their solutions over the course of the competition. **Our winner is [Yash Kant](https://yashkant.github.io/).** We chose Yash for learning reusable digit operations and composing them across repeated computation. The calculator's wiring is supplied, so this is a narrower result than learning the entire algorithm. Congrats, Yash!

## Honourable mention

**alirezashirvani-jr (`f5d75083`) reached 10.38% aggregate Hard accuracy.** It represents numbers as positions on clocks with different periods, learns transitions between those positions, and combines learned votes into an integer answer. Its larger-number branch uses learned Fourier operators that can be composed at evaluation. The transitions are learned rather than computed by an explicit squaring formula. Although it failed T=1 certification, its learned transitions distinguish it from the top leaderboard submissions, and its results are promising.

## Interesting ideas that didn't quite work

None of these submissions passed all the single-step (T=1) tests on Hard, but research is about much more than benchmark climbing.

| Submission | Implemented idea |
|---|---|
| AHappyCPU (`1f7c46f2`) | Gradually replaces continuous recurrent states with discrete choices from a learned 64-entry codebook. |
| Ethan Davis (`24c58fc1`) | Uses N to generate convolution kernels that move information around a cyclic tape, making the update operation depend on the modulus. |
| Chris Wood (`7f055799`) | Learns a five-instruction program from 16 supplied integer operations by comparing program outputs with labels. The interpreter is fixed; instruction composition is learned. |
| Rohan Sehgal (`3854d99a`) | Learns wiring and add/multiply choices for three parallel gates per expert, with exact integer forward values and approximate gradients. Modular reduction is supplied. |
| Aqua (`55bec2ea`) | Sends learned digit-pair interactions to fixed decimal places `i+j`, then combines affine updates with a parallel scan. The arithmetic layout is supplied. |

## Next steps

Current submissions appear to learn wiring around Python primitives. Could we learn the calculator's wiring as well as its digit operations? The question is whether those parts transfer to unfamiliar algorithms and longer executions when trained end to end.

All 15,602 uploads are preserved in the [private GPU MODE dataset on Hugging Face](https://huggingface.co/datasets/GPUMODE/one-layer-deeper-submissions) for further analysis.
