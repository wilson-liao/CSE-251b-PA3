
import torch
import torch.nn as nn
import numpy as np

# Text generation function
def generate_text(model, device, char_idx_map, idx_to_char, max_len=1000, temp=0.8):
    model.eval()
    start_text = "Macbeth\n by William Shakespeare\n Edited by Barbara A. Mowat and Paul Werstine"
    input_seq = [char_idx_map[c] for c in start_text]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
    
    model.to(device)
    generated_text = start_text
    for _ in range(max_len):
        with torch.no_grad():
            output = model(input_tensor)
            output = output.squeeze().div(temp).exp()
            predicted_idx = torch.multinomial(output, 1).item()
            predicted_char = idx_to_char[predicted_idx]
            generated_text += predicted_char
            input_tensor = torch.cat((input_tensor[:, 1:], torch.tensor([[predicted_idx]], dtype=torch.long).to(device)), dim=1)
    
    return generated_text


def generate_heatmap(generated_song, heatmap, neuron_idx=0):
    """
    Generates a heatmap using the provided generated song, heatmap chart and neuron id.

    Parameters:
    - generated_song (nn.Module): The song generated by a trained model.
    - heatmap (torch.Tensor): heatmap/activation values from a particular layer of the trained model.
    - neuron_idx (int): id of the neuron to plot heatmap for.

    Returns:
        None
    """
    pad_factor = 20
    heatmap = heatmap.detach().numpy()

    data = np.append(heatmap[:,neuron_idx], 0.0)
    padded_song, padded_data = pad(generated_song, data, pad_factor=pad_factor)

    padded_song = np.reshape(padded_song, (len(padded_song)//pad_factor, pad_factor))
    padded_data = np.reshape(padded_data, (len(padded_data)//pad_factor, pad_factor))

    plt.figure(figsize=(heatmap.shape[0]//4,heatmap.shape[1]//4))
    plt.title(f"Heatmap For Song RNN, Neuron ID: {neuron_idx}")
    heatplot = plt.pcolor(padded_data, edgecolors='k', linewidths=4, cmap='RdBu_r', vmin=-1.0, vmax=1.0)

    show_values(heatplot, song=padded_song)
    plt.colorbar(heatplot)
    plt.gca().invert_yaxis()
    plt.savefig(f"./plots/heatmap_{neuron_idx}.png")
    print(f"==> Heatmap saved for Neuron ID: {neuron_idx}..")
    return