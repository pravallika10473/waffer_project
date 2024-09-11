import tensorflow as tf
import os

def test_gpu():
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Is built with CUDA: {tf.test.is_built_with_cuda()}")
    print(f"CUDA version: {tf.sysconfig.get_build_info()['cuda_version']}")
    print(f"cuDNN version: {tf.sysconfig.get_build_info()['cudnn_version']}")
    print(f"Is GPU available: {tf.config.list_physical_devices('GPU')}")

    print("\nEnvironment variables:")
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")

    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print("\nGPUs are available:")
        for gpu in gpus:
            print(f"- {gpu}")
    else:
        print("\nNo GPUs detected by TensorFlow.")

    # Create a simple TensorFlow computation
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        print("\nRunning a sample computation on", 'GPU' if gpus else 'CPU')
        # Create random tensors
        a = tf.random.uniform((1000, 1000))
        b = tf.random.uniform((1000, 1000))
        
        # Perform matrix multiplication
        c = tf.matmul(a, b)
        
        print("Computation completed.")
        print("Result shape:", c.shape)

if __name__ == "__main__":
    test_gpu()