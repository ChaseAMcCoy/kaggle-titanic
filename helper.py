import numpy as np

def score_cabin(cabin: str):
    """
    Scores a given cabin based on how high its deck is.

    Parameters:
    cabin (str): A cabin identifier.

    Returns:
    int: The score assigned to the cabin.
    """

    deck = cabin[0]
    if deck == 'A':
        return 6
    elif deck == 'B':
        return 5
    elif deck == 'C':
        return 4
    elif deck == 'D':
        return 3
    elif deck == 'E':
        return 2
    elif deck == 'F':
        return 1
    elif deck == 'G':
        return 0
    elif deck == 'T':
        return 7
    else:
        return 3
        
def score_cabins(cabins: list[str]):
    """
    Calculate the average score for a list of cabins.
    
    Parameters:
    cabins (list[str]): A list of cabin names.
    
    Returns:
    float: The average score for the cabins.
    """
    return np.mean([score_cabin(cabin) for cabin in cabins])
