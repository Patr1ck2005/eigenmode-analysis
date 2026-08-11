# Candidate data selectors

Tracking APIs may receive auxiliary candidate data independently of the
features used for assignment. A selector maps each candidate identity to the
auxiliary record that should be carried into the result.

Selectors must be deterministic, preserve explicit candidate identity, and
return missing values when no valid candidate exists. They must not infer
identity from equal floating-point values or from array position after an
assignment has been performed.

Adapters may translate table, manifest, or simulator payloads into this
selector contract. The shared package remains schema-neutral.
