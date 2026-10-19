def dense_linear_ops(batch, sequence, input_dim, hidden_dim, output_dim):
    # Multiply-add proxy for a dense temporal model.
    return batch * sequence * (input_dim * hidden_dim + hidden_dim * output_dim)

def snn_active_synops(input_spikes, hidden_dim, output_spikes, memory_dim):
    # Conceptual active-synapse proxy. Not actual wall-clock compute on dense PyTorch kernels.
    return input_spikes * hidden_dim + output_spikes * (hidden_dim + memory_dim)
