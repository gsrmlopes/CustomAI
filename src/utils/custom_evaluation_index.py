def custom_evaluation_index(accuracy, precision, recall):
    """
    The function calculates a custom evaluation index using accuracy, precision, and recall.
    
    :param accuracy: The accuracy parameter represents the proportion of correctly classified instances
    out of the total number of instances. It is calculated as the sum of true positives and true
    negatives divided by the sum of true positives, true negatives, false positives, and false negatives
    :param precision: Precision is a measure of how many correctly predicted positive instances out of
    all instances predicted as positive. It is calculated as the ratio of true positives to the sum of
    true positives and false positives
    :param recall: Recall is a measure of how well a model can identify all the relevant instances in a
    dataset. It is calculated as the ratio of true positive predictions to the sum of true positive and
    false negative predictions
    :return: the custom evaluation index, which is calculated using the accuracy, precision, and recall
    values.
    """
    return (2 * accuracy * precision * recall) / (accuracy + precision + recall)
