import numpy as np
import matplotlib.pyplot as plt

def display_some_examples(examples, labels):
    """
    Display the dataset examples (train/test)
    
    Parameters:
        examples: Train or test dataset (numpy array - uint8)
        labels: Train or test labels (numpy array - uint8)

    Returns:
        None
    """

    plt.figure(figsize = (10, 10))

    for i in range(25):
        # select a random index in the dataset
        idx = np.random.randint(0, examples.shape[0] - 1)
        img = examples[idx]
        label = labels[idx]

        # subplot the dataset examples
        plt.subplot(5, 5, i + 1)
        plt.title(f'Label: {label}')
        plt.tight_layout(rect = [0, 0, 1, 0.95])
        plt.imshow(img, cmap = 'gray')

    plt.suptitle("Dataset Examples")
    plt.show()