def calculate_f_0_5(true_pairs, pred_pairs):
    """
    true_pairs: Set of tuples (source1_id, source2_id) from ground truth
    pred_pairs: Set of tuples (source1_id, source2_id) from your model
    """
    # Convert to sets if they aren't already
    true_set = set(true_pairs)
    pred_set = set(pred_pairs)
    
    # Calculate True Positives, False Positives, False Negatives
    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    # Precision and Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F0.5 Score Formula
    beta = 0.5
    if (precision + recall) == 0:
        return 0.0
        
    f_0_5 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    
    return f_0_5

# --- Test the function ---
if __name__ == "__main__":
    # Example from the competition docs
    truth = {("A", "1"), ("A", "2"), ("B", "3")}
    prediction = {("A", "1"), ("A", "2"), ("C", "4")}
    
    score = calculate_f_0_5(truth, prediction)
    print(f"F0.5 Score: {score:.4f} (Should be close to 0.714)")