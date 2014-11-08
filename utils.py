def get_prediction_stats(predicted_targets, real_targets):
    assert len(predicted_targets) == len(real_targets)
    amount_of_targets = len(real_targets)
    incorrectly_predicted = []
    for predicted, real in zip(predicted_targets, real_targets):
        if predicted != real:
            incorrectly_predicted.append({"real": real, "predicted": predicted})

    amount_incorrect = len(incorrectly_predicted)
    amount_correct = amount_of_targets - amount_incorrect
    accuracy = 1 - amount_incorrect/float(amount_of_targets)

    print ("{0} out of {1} targets were predicted correctly "
           "({2}% correct).".format(
        amount_correct, amount_of_targets, accuracy * 100))
    return accuracy, incorrectly_predicted, amount_of_targets
