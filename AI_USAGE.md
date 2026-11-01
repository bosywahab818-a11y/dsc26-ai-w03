# AI usage

## Tools I used, and for what

- **ChatGPT:** Helped implement the scikit-learn pipeline, customer history features, chronological time split, and evaluation functions. I reviewed the implementation and ran the tests and experiments myself.
- **PowerShell:** Used to run commands and inspect the project.
- **uv:** Used to install project dependencies and run the test suite and experiments.
- **pytest:** Used to run the public tests and verify the implementation.

## AI-suggested code I rejected or fixed in add_history_features or time_split

I did not identify a specific AI-suggested code block that needed to be rejected or fixed. However, I specifically verified two important edge cases: rows with the same customer and timestamp must not count as prior history, and the chronological split must keep all rows with the same boundary timestamp in the test set. My own checks confirmed both behaviors, as well as that history features are independent of input row order.

## How I checked that my model has no leak

I excluded post-creation columns from the features, calculated customer history using only strictly earlier timestamps, used a chronological train/test split, and fitted the model only on training data before evaluating it on the test set.