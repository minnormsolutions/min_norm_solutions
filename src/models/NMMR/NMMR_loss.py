from src.models.NMMR.kernel_utils import calculate_kernel_matrix_batched


def NMMR_loss(model_output, target, kernel_matrix, loss_name: str, l2_lambda = 0):
    """Computes the U or V-statistic loss."""
    
    residual = target - model_output
    n = residual.shape[0]
    K = kernel_matrix

    if loss_name == "U_statistic":
        # calculate U statistic (see Serfling 1980)
        K.fill_diagonal_(0)
        unpenalized_loss = (residual.T @ K @ residual) / (n * (n-1))
    elif loss_name == "V_statistic":
        # calculate V statistic (see Serfling 1980)
        unpenalized_loss = (residual.T @ K @ residual) / (n ** 2)
    else:
        raise ValueError(f"{loss_name} is not valid. Must be 'U_statistic' or 'V_statistic'.")

    norm2 = (model_output.T @ model_output) / n

    if l2_lambda:
        penalized_loss = unpenalized_loss + l2_lambda * norm2
    else:
        penalized_loss = unpenalized_loss

    return penalized_loss[0,0], unpenalized_loss[0,0], norm2[0,0]


def NMMR_loss_batched(model_output, target, kernel_inputs, kernel, batch_size: int, loss_name: str,
                        l2_lambda = 0):
    """Computes the U or V-statistic loss in blocks to limit memory usage."""
    # Blocks are batch_size x sample_size.

    residual = target - model_output
    n = residual.shape[0]

    unpenalized_loss = 0
    norm2 = 0
    for i in range(0, n, batch_size):
        partial_kernel_matrix = calculate_kernel_matrix_batched(kernel_inputs, (i, i+batch_size), kernel)
        if loss_name == "U_statistic":
            # zero out the main diagonal of the full matrix
            for row_idx in range(partial_kernel_matrix.shape[0]):
                partial_kernel_matrix[row_idx, row_idx+i] = 0
        temp_loss = residual[i:(i+batch_size)].T @ partial_kernel_matrix @ residual
        temp_norm2 = ( model_output[i:(i+batch_size)].T @ model_output[i:(i+batch_size)] )
        unpenalized_loss += temp_loss[0,0]
        norm2 += temp_norm2[0,0]
    
    if loss_name == "U_statistic":
        unpenalized_loss /= (n * (n-1))
    elif loss_name == "V_statistic":
        unpenalized_loss /= (n ** 2)
    norm2 /= n
    if l2_lambda:
        penalized_loss = unpenalized_loss + l2_lambda * norm2
    else:
        penalized_loss = unpenalized_loss

    return penalized_loss, unpenalized_loss, norm2
