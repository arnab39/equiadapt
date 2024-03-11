# Commands to run N-Body Model
```
python examples/nbody/train.py
```
# Changing Prediction
In `examples/nbody/train.py`, change the value for `HYPERPARAMS["pred_model_type"]`. Currently, the options are the following: GNN, EGNN, vndeepsets, and Transformer.

# Changing Canonicalizer
Currently only vndeepsets is supported.