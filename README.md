Creating an Accelerator-Friendly CNN
# Files
- ConvNet.py: Defines the model
- CNN.py: Train the model, MNIST dataset for image classifcation
- quantize.py: Simulating quantization effects, `num_bits` represents the quantization bits
    - usage: python quantize.py --print 0 --num-bits 8
----------------------------------------------------
- param_sweep.sh: Hyperparameter sweeping using multiple GPUs; run CNN.py
    - including `batch_size` and `learning_rate`
- run_cnn.sh: Train the model; run CNN.py
- run_convnet.sh: Load and test different models; Plot histograms
