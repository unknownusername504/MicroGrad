import numpy as np


def assert_array_almost_equal(a, b, atol=1e-6):
    # Check if the dtype is the same
    assert a.dtype == b.dtype, f"Data types do not match: a:{a.dtype} != b:{b.dtype}"
    # Check if the shape is the same
    assert a.shape == b.shape, f"Shapes do not match: a:{a.shape} != b:{b.shape}"
    # Check if the values are the same
    assert np.allclose(a, b, atol=atol), f"Values do not match: a:{a} != b:{b}"


def validate_z_values(observed_z, z_torch, cast_dtype):
    observed_z_np = observed_z.get_value()
    z_torch_np = z_torch.detach().numpy()
    if cast_dtype != None:
        z_torch_np = z_torch_np.astype(cast_dtype)

    # Validate values
    assert_array_almost_equal(observed_z_np, z_torch_np)


def validate_gradients(tensor_pairs):
    # Validate gradients
    for t, t_torch in tensor_pairs:
        assert_array_almost_equal(t.grad, t_torch.grad.numpy())
