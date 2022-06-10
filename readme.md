# Modifications:

## 1. Preprocessing part:

This part is responsible for extracting frames from videos, then convert and save images (from two dataset) as numpy array.

But **loading all the images, resizing, converting and saving them array** was taking too long and using **single thread** of processor. I've ultilized **multi-threading** to make this process faster as modified code used **all four cores.**

## 2. Model Training part:

* The author tried to used model checkpoint, but it wasn't utilized during model training. The reason was, author set monitor "mean_squared_error" but during model compilation, metrics was just "accuracy". I've added "mean_squared_error" in metrics along with accuracy. It solved that issue.

* The author set to monitor "val_loss" for early stopping eventhoug there wasn't any "val_loss" for this model. I've changed it to 'mean_squared_error' to solve this issue.
* I've added ReduceLROnPlateau to reduce learning rate in real-time.

## 3. Model testing part:
* If anomaly is detected at any frame, those frames will automatically stored in anomal_images folder
