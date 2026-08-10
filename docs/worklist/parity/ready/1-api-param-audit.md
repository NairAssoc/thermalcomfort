# Audit Rust signatures against Python parameter lists

For every ported function, does the Rust signature expose every parameter Python
accepts — including `**kwargs`-style optionals?

Found while deciding the sleep model's shape: `two_nodes_gagge_ji` hardcodes
`length_time_simulation = 120` (`src/models/two_nodes_gagge.rs:942`) where Python takes it
as a kwarg, and does not expose `body_weight`, `initial_skin_temp` or `initial_core_temp`
at all. A hardcoded constant standing in for a Python parameter is a parity gap: the
caller cannot ask a question Python answers, and no parity test can catch it because the
sweep never varies what the Rust API cannot express.

Governing principle (user, this branch): **match the basic Python API, modulo our newtypes,
for both input and output.** We are porting functionality, not redesigning it; parity
testing only works when the calls are the same. Ownership/borrowing is ours to choose.

Output: a table of function → Python params → Rust params → gap. Feeds item 2.
