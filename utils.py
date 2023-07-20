import torch

def process_caption(caption, max_caption_length=200, alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "):
    """
    Converts a caption string to a tensor of one-hot encoded vectors based on the given alphabet.

    Args:
        caption (str): The input caption to be converted to one-hot encoded vectors.
        max_caption_length (int): The maximum length of the output label sequence. If the caption is longer, it will be truncated.
        alphabet (str, optional): The alphabet used to map characters to numeric labels and define the length of the one-hot vectors. Default is a combination of lowercase letters, numbers, and common punctuation marks.

    Returns:
        torch.Tensor: A tensor containing the one-hot encoded vectors for the caption.
    """
    # Convert the caption to lowercase for case-insensitivity
    caption = caption.lower()

    # Create a mapping from characters in the alphabet to numeric labels
    alpha_to_num = {k: v + 1 for k, v in zip(alphabet, range(len(alphabet)))}

    # Initialize the output tensor with zeros and set the data type to long
    labels = torch.zeros(max_caption_length).long()

    # Determine the maximum number of characters to process from the caption
    max_i = min(max_caption_length, len(caption))

    # Convert each character in the caption to its corresponding numeric label
    for i in range(max_i):
        # If the character is not in the alphabet, use the numeric label for space (' ')
        labels[i] = alpha_to_num.get(caption[i], alpha_to_num[' '])
    
    labels = labels.unsqueeze(1)
    
    # Convert the numeric labels to one-hot encoded vectors
    # Initialize a tensor of zeros with the shape (sequence length, alphabet length + 1) and scatter ones based on the labels
    one_hot = torch.zeros(labels.size(0), len(alphabet) + 1).scatter_(1, labels, 1.)
    
    # Remove the column corresponding to the numeric label 0 (used for padding)
    one_hot = one_hot[:, 1:]
    
    # Permute the tensor to have the sequence length as the first dimension
    one_hot = one_hot.permute(1, 0)

    return one_hot

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)