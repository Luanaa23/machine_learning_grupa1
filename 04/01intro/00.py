import tensorflow as tf

tensor_od = tf.constant(42)
print(f"Tensor 0D (scalar): {tensor_od.numpy()}")

tensor_1d = tf.constant([1, 2, 3, 4])
print(f'Tensor 1D (vector): {tensor_1d.numpy()}')

tensor_2d = tf.constant([[1,2,3], [4,5,6]])
print(f'Tensor 2D (Matrice): {tensor_2d.numpy()}')

tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f'Tensor 3D (Matrice): {tensor_3d.numpy()}')

