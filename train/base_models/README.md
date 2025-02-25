# Base Models
Model files containing the trained parameters
for many of the base models used are under the
[base_model_trained_files](../base_model_trained_files) directory.
Others are too large to be tracked on git -- ask Jack for access to
the s3 bucket that they are stored on.

### Adding a new base model
If you'd like to add a new base model for evaluation, you need to make to
perform the following steps:
1. Implement the base model in PyTorch. Your base model must inherit from
   `torch.nn.Module` and must implement the `__init__` and `forward` methods.
   The `forward` method must take just one parameter, a batch of samples over
   which a forward pass is performed.
2. Train your base model and save the base model's state dictionary to a file.
   This may be done using `torch.save(my_base_model.state_dict(), "model.t7")`
3. Create a new entry in [train_config.py](../train_config.py) under `get_base_model` and `get_parity_model`. Note the following requirements: 
   1. `base_path`: Specify the path to the state dictionary saved in step (2).
   2. `class`: Specify the classpath of your base model in the "class" field
   and any arguments required for the `__init__` function in the "args" field.
   3. `input_size`: Specify the input dimensions expected of inputs to
   the `forward` method of your base model.
