### 1.a. Inputs and Outputs of the Pre-trained Models (Feature Extraction)

In this phase, the pre-trained backbones (**Omnivore** and **SlowFast**) function as encoders to transform raw video data into compact mathematical representations.

* **INPUT (Video Sub-segments):**
* The models do not process the entire video at once. Instead, the input consists of **short, fixed-duration video sub-segments** (typically **1-second snippets**) derived from the raw video.

* Technically, this input is a tensor of stacked **RGB frames** corresponding to that specific time window within a recipe step.

* **OUTPUT (Feature Vectors/Embeddings):**
* The output is a **high-level feature vector** (embedding) for each sub-segment.

* This is a fixed-dimensional array of floating-point numbers that summarizes the semantic content (visual cues, motion, objects) of that specific second of video, discarding redundant pixel-level data.

* **Result:** For a single recipe step, the final output is a **sequence of these vectors** , where each vector represents one sub-segment. This sequence serves as the direct input for your  (MLP) and  (Transformer) baselines.

Based on the error analysis JSON files you provided and the project requirements, here is the answer for **Step 2.a** (Analysis of model performance on different error types) in English.

### 2.a. Performance Analysis by Error Type

We reproduced the  (MLP) and  (Transformer) baselines using the `Omnivore` backbone. Below is a detailed analysis of their performance across different error categories based on the experimental results.

#### **1. Overall Comparison ( vs. )**

* **Step-Level Analysis:**
The **MLP ()** baseline generally outperforms the **Transformer ()** on the step-level task.
* 
**MLP Global Accuracy:** ~56.8% vs. **Transformer Global Accuracy:** ~46.7%.


* 
**MLP Global F1:** ~0.55 vs. **Transformer Global F1:** ~0.48.
The MLP demonstrates a higher Recall (~0.86) compared to the Transformer (~0.79), suggesting it is more sensitive to detecting errors, though both struggle with precision.




* **Recording-Level Analysis:**
At the recording level, the trend is mixed. While the Transformer achieves higher global accuracy (~57.1% vs ~44.3%), it suffers from a significantly lower Recall (~0.49) compared to the MLP (~0.86). This indicates the Transformer misses more than half of the actual errors, whereas the MLP detects most errors but generates more false positives.



#### **2.a Analysis by Specific Error Types**

Both models show a distinct pattern where they perform significantly better on specific "overt" error types compared to the "No Error" (Normal) class.

* **High Performance Categories:**
Both models achieve high accuracy (often > 80%) on explicit error types such as **Technique**, **Preparation**, **Measurement**, and **Temperature** errors.


* 
*Example (MLP Step-Level):* **Measurement Error** (Acc: ~90.5%, F1: 0.95) and **Technique Error** (Acc: ~87.1%, F1: 0.93) are detected with very high reliability.


* This suggests that the visual features extracted by Omnivore are discriminative enough to capture significant deviations in action execution (e.g., using the wrong tool or incorrect motion).


* **The "No Error" Bottleneck:**
The primary struggle for both models is correctly classifying the **"No Error"** (normal execution) class.
* 
**MLP (Step-Level):** Accuracy drops to ~51.7% for "No Error" samples.


* 
**Transformer (Step-Level):** Accuracy is even lower at ~41.2%.


* This low performance on the majority class ("No Error" has ~691 samples vs. <70 for error classes) significantly drags down the global metrics. The models tend to over-predict errors (high Recall for error classes, but low Precision globally), leading to many false positives where normal actions are flagged as mistakes.



#### ** Conclusion**

The **MLP baseline ()** currently provides a more robust starting point than the Transformer () for this specific feature set, offering better recall and F1 scores. The high performance on specific error categories is promising, but future improvements must focus on better distinguishing "Normal" steps to reduce the false positive rate.


The code for the **RNNBaseline** is correct and well-structured for the required task. You have implemented exactly what was suggested in point **2.b** of the project ("*Propose a new baseline... For example, you could train an RNN/LSTM*").

Here is the translation of the checklist and next steps to ensure you have everything needed for the report:

## 2.b. New Baseline: RNN/LSTM
### 1. Technical Details (Check before training)

* **Loss Function:** Since your model returns a linear output (`self.fc`) without a final activation function (like Sigmoid), make sure to use **`nn.BCEWithLogitsLoss()`** during training. If you were to use `nn.BCELoss()`, you would need to add `torch.sigmoid(out)` at the end of the `forward` method.
* **Input Dimension:** When instantiating the class, `input_dim` must match the dimension of the features you are using (e.g., **1024** for Omnivore or **2304** for SlowFast, as seen in the feature analysis).

### 2. What you need for the Report (Step 2.b)

To complete this section of the report, you now need to:

1. **Train the model:** Run the training on the same data used for the MLP and Transformer.
2. **Generate results:** Produce the JSON file with the metrics (Accuracy, F1, Precision, Recall) similar to the ones you showed me earlier (`MLP_omnivore_step_error_analysis.json`).
3. **Compare:** You must comment on how this LSTM performs compared to the provided baselines ( and ).
* *Hypothesis:* The LSTM should theoretically handle the sequential nature of videos better than the MLP (which treats segments in isolation or aggregates them simply).



### 3. Text for the Report (Draft)

If you need a base text to describe the model in the report, you can use this draft:

> **Proposed Baseline (LSTM):**
> To better capture the temporal dependencies between video snippets within a recipe step, we implemented a Recurrent Neural Network based on Long Short-Term Memory (LSTM) units. Unlike the MLP baseline (), which processes aggregated features, the LSTM processes the sequence of frame features step-by-step.
> * **Architecture:** The model consists of an LSTM layer followed by a linear classification head.
> * **Input:** A sequence of feature vectors of dimension  (corresponding to the backbone output size).
> * **Mechanism:** We utilize the hidden state of the last time step () as a compact summary of the entire action sequence, which is then fed into the classifier to predict the error probability.
> 
> 

**Next Step:** Once you have the results (the JSON or the metrics), feel free to share them here. I can help you write the final comparative analysis between the MLP, Transformer, and your new LSTM.