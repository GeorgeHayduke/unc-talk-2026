# Data Science Interview Questions: Model Answers

Each question has a 30-second answer to say aloud and a deeper version for follow-ups.

## 1. Statistics and ML fundamentals

### What is a p-value?

**30 seconds:** A p-value is the probability of seeing a result at least as extreme as mine if the null hypothesis were true. It is not the probability that the null is true, and it says nothing about how large or important the effect is.

**Go deeper:** Report it with an effect size and a confidence interval. Large samples make trivial effects significant, and testing many hypotheses inflates false positives (correct with Bonferroni or false discovery rate). Example: in an A/B test, p = 0.03 means that if there were truly no difference, a gap this big would show up about 3% of the time.

### What is the difference between logistic and linear regression?

**30 seconds:** Linear regression predicts a continuous outcome by minimizing squared error. Logistic regression predicts the probability of a binary outcome by modeling the log-odds as a linear function of the features, fit by maximum likelihood.

**Go deeper:** Logistic coefficients are changes in log-odds; exponentiate them for odds ratios. Linear regression on a 0/1 outcome can predict values outside 0 to 1 and violates its error assumptions. Logistic regression outputs probabilities and becomes a classifier only once you choose a threshold. Both take regularization (L1, L2).

### What is K-fold cross-validation and what is it used for?

**30 seconds:** Split the data into k folds, train on k-1 of them, validate on the held-out fold, repeat so each fold is held out once, and average the scores. It estimates how well a model generalizes and supports model selection and hyperparameter tuning, especially with limited data.

**Go deeper:** Use stratified folds for imbalanced classes and group folds when rows from one entity (a customer, a patient) must stay together. For time series, use forward-chaining splits, never shuffled folds. Keep a final test set untouched, and use nested cross-validation for an honest estimate when tuning heavily. Five or ten folds is typical.

### How can you tell if your model is overfitting?

**30 seconds:** Training performance keeps improving while validation performance stalls or gets worse. A large gap between the two is the signature.

**Go deeper:** Plot learning curves and validation curves (error vs. model complexity or boosting rounds). Fixes: more data, regularization, a simpler model, early stopping, pruning, feature selection. If results look too good, suspect leakage before celebrating, and confirm on a time-based holdout. A model that degrades sharply in production is often overfit to its training window.

### Given a classification model, how do you interpret a ROC chart?

**30 seconds:** A ROC curve plots true positive rate against false positive rate across every threshold. The diagonal is random guessing, and the closer the curve hugs the top-left corner the better. AUC is the probability that a random positive is scored above a random negative: 0.5 is chance, 1.0 is perfect.

**Go deeper:** Each point is a threshold, so choose one from the cost of errors, not from AUC. With a very rare positive class, ROC can look flattering because the false positive rate has a huge denominator; also check the precision-recall curve. AUC measures ranking, not calibration.

### What is the difference between precision and recall?

**30 seconds:** Precision is the share of flagged cases that are truly positive (TP / (TP + FP)). Recall is the share of all true positives that I caught (TP / (TP + FN)). Raising the threshold usually trades recall for precision.

**Go deeper:** Pick by cost. If reviewing each flag is expensive (fraud investigators), favor precision. If a miss is costly (disease screening), favor recall. F1 is their harmonic mean; F-beta weights recall more or less. Report precision at a fixed recall, or recall at a fixed precision, when the business has a capacity limit.

## 2. Rare events, calibration, and explainability

### The class occurs less than 0.1% of the time. What can you do to improve performance, and how do you make sure the model generalizes?

**30 seconds:** First change how I measure: accuracy is useless here, so I use precision-recall AUC and recall at a fixed precision. Then I improve training with class weights (for example scale_pos_weight in XGBoost), undersampling the majority class, or careful oversampling, and I tune the decision threshold. To generalize, I split before any resampling, keep the test set at the true base rate, use stratified or time-based splits, and check that each fold has enough positives.

**Go deeper:**

- Resample only inside training folds. Oversampling before the split leaks copies of the same positive into validation.
- Reweighting or resampling distorts predicted probabilities, so recalibrate on data at the true base rate if the probabilities matter.
- Count positives, not rows: 0.1% of one million rows is about 1,000 positives, and that is your real sample size.
- Get more positives where possible: label collection, active learning, or a partner's data. Add domain features that separate the rare class.
- If labels are scarce or delayed, consider anomaly detection as a first filter.
- Monitor precision and drift after launch, since the base rate can move.

### What is model calibration? Why is it important, and what techniques have you used?

**30 seconds:** A model is calibrated when its predicted probabilities match observed frequencies: among cases scored 0.2, about 20% are actually positive. It matters whenever a probability drives a decision, such as pricing, expected-cost thresholds, risk scores, or combining models, and after reweighting for imbalance.

**Go deeper:** Diagnose with a reliability diagram plus Brier score, log loss, or expected calibration error. Fix with Platt scaling (a sigmoid fit), isotonic regression (more flexible, needs more data), or temperature scaling for neural networks, always fit on held-out data, not the training set. Random forests and boosted models trained with non-log losses are often miscalibrated; log-loss gradient boosters are usually closer but still worth checking. Add which of these you have used, with a real example, before the interview.

### What are partial dependence, ICE, and ALE plots, and what are they used for?

**30 seconds:** All three show how a feature affects a model's predictions. A partial dependence plot (PDP) shows the average prediction as one feature varies. An ICE plot draws one line per observation, so you can see where individuals differ from the average. ALE plots average the local change in prediction within small intervals of the feature and accumulate it.

**Go deeper:** PDPs can hide interactions, because opposite effects average out; ICE reveals them. PDPs also average over unrealistic combinations when features are correlated; ALE avoids this by only using real neighboring data, so prefer it with correlated inputs. Use these plots to check direction and shape of an effect against domain knowledge, alongside SHAP dependence plots. None of them shows causation.

## 3. Tree models

### What is a random forest, and how does it work?

**30 seconds:** A random forest is an ensemble of decision trees. Each tree trains on a bootstrap sample of the rows and considers only a random subset of features at each split. Predictions are averaged (regression) or voted (classification). Averaging many decorrelated, deep trees cuts variance without adding much bias.

**Go deeper:** The rows each tree never saw give a free out-of-bag error estimate. Trees train independently, so the forest parallelizes easily and is a robust default with little tuning. Built-in impurity importance is biased toward high-cardinality features; prefer permutation importance or SHAP.

### How is it different from gradient boosting or XGBoost?

**30 seconds:** A random forest builds deep trees independently and averages them, which reduces variance (bagging). Gradient boosting builds shallow trees one after another, each fit to the errors of the ensemble so far, which reduces bias (boosting). XGBoost is a fast, regularized implementation of gradient boosting.

**Go deeper:**

- XGBoost adds L1 and L2 penalties on leaf weights, uses second-order gradient information, handles missing values natively, and supports row and column subsampling.
- CatBoost uses ordered boosting and native categorical encoding, which reduces target leakage on categorical features. LightGBM grows trees leaf-wise and is very fast on large data.
- Boosting usually wins on tabular accuracy but is more sensitive to hyperparameters and overfits without early stopping and a small learning rate. Forests are harder to break.
- Both pair well with TreeSHAP for explanation.

## 4. SQL

### Top 10 customers by order count. Does anything change for dollar amount?

**By order count:** join, group by the customer's ID, count orders, sort, and take ten.

```sql
SELECT c.customer_id,
       c.customer_name,
       COUNT(o.order_id) AS order_count
FROM   customer c
JOIN   orders o ON o.customer_id = c.customer_id
GROUP  BY c.customer_id, c.customer_name
ORDER  BY order_count DESC, c.customer_id
LIMIT  10;
```

**By dollar amount:** the aggregate changes from a count to the sum of each line's total, quantity times price.

```sql
SELECT c.customer_id,
       c.customer_name,
       SUM(o.qty * o.price) AS total_amount
FROM   customer c
JOIN   orders o ON o.customer_id = c.customer_id
GROUP  BY c.customer_id, c.customer_name
ORDER  BY total_amount DESC
LIMIT  10;
```

**Things to say out loud:**

- Group by customer ID, not name, because names can repeat.
- Use `SUM(qty * price)`, not `SUM(price)`; price is per unit.
- The ranking can change with the metric. In the sample data, Bill de Blasio has the most orders (4) but Hillary Clinton has the highest total ($2,753.00 vs. $2,326.00).
- An inner join drops customers with no orders; use a left join if they matter.
- Ties: `LIMIT` cuts arbitrarily, so use `DENSE_RANK()` in a subquery if all tied customers should appear.
- Ask about returns, cancellations, discounts, and currency before trusting the total.
- Syntax varies: `TOP 10` in SQL Server, `FETCH FIRST 10 ROWS ONLY` in Oracle.

### Find email addresses in common between AWS and Amazon customers, and those in AWS but not in Amazon.

**In common:**

```sql
SELECT DISTINCT LOWER(TRIM(a.email)) AS email
FROM   aws_customers a
JOIN   amazon_customers m
       ON LOWER(TRIM(a.email)) = LOWER(TRIM(m.email));
```

**In AWS but not in Amazon:**

```sql
SELECT DISTINCT LOWER(TRIM(a.email)) AS email
FROM   aws_customers a
WHERE  NOT EXISTS (
         SELECT 1
         FROM   amazon_customers m
         WHERE  LOWER(TRIM(m.email)) = LOWER(TRIM(a.email)));
```

**Things to say out loud:**

- `INTERSECT` and `EXCEPT` (`MINUS` in Oracle) give the same results with less code, on normalized values.
- Normalize first: emails differ by case and stray whitespace.
- Avoid `NOT IN` with a subquery that can contain NULLs: one NULL makes it return no rows. `NOT EXISTS` or a left join with `IS NULL` is safe.
- `DISTINCT` guards against customers listed more than once.
- On large tables, join on a normalized, indexed column rather than wrapping functions around both sides.

## 5. Case questions

For both, ask before you answer, then walk a fixed path: clarify, define labels, data, baseline, model, evaluate, act, monitor.

### A customer onboards about 1,000 new accounts a day (username, password, email, IP address, user agent) and is worried about fraudsters. How would you approach it?

**Key questions to ask:**

- What does fraud look like here: fake accounts, bots, promotion abuse, later account takeover?
- What does a fraudster cost, and what does blocking a real customer cost?
- Are there labels: confirmed fraud, chargebacks, manual review outcomes? How long until they arrive?
- What actions exist: block, extra verification, send to review? How many reviews can the team handle?
- Must the decision happen in real time at signup?

**30 seconds:** I would start with the cost of each kind of error and the labels available. Then build a baseline of rules and velocity checks (accounts per IP, device, or email domain per hour), engineer features from the email (domain, disposable providers, patterns), IP (geography, proxy or datacenter, reputation), and user agent (headless or spoofed clients). If there are labels, train a gradient boosted model with rare-event handling; if not, use anomaly detection and link analysis on shared devices and IPs. The output is a risk score with tiered actions: allow, step-up verification, review, block.

**Go deeper:** Validate on a time-based split, measure precision at the review team's capacity and recall on confirmed fraud, and A/B test the actions. At 1,000 signups a day, positives will be few, so label quality and feedback from reviewers matter more than model complexity. Fraudsters adapt, so monitor drift and retrain. Keep the model explainable enough for analysts and customer support.

### A customer wants to ensure that images and text their users post are not offensive. How would you approach it?

**Key questions to ask:**

- What exactly counts as offensive: hate, harassment, sexual content, violence, self-harm? Who writes and owns the policy?
- Must content be blocked before publishing, or reviewed after?
- What volume, latency, languages, and regions?
- How costly is wrongly blocking a post versus letting harm through?
- What human review capacity and appeals process exist? Is there labeled data?

**30 seconds:** I would turn the policy into a taxonomy with severity levels, then start with pretrained moderation models or an LLM-based classifier plus blocklists as a baseline, since that is fast and measurable. Images with text need a multimodal approach. Each category gets its own thresholds and tiered actions: auto-block high confidence, human review in the middle, allow the rest.

**Go deeper:** Build a labeled evaluation set with clear guidelines and check inter-annotator agreement. Evaluate precision and recall per category, language, and community, since context, slang, and sarcasm cause uneven errors. Plan for adversarial evasion (misspellings, altered images), feed reviewer decisions back into training, and monitor drift. Account for privacy and the wellbeing of human reviewers.

## 6. Behavioral questions

These need your own real stories, so each has a scaffold instead of a scripted answer. Budget your time: Situation and Task 10-15%, Action about 60%, Result the rest, quantified, plus what you learned.

### What is the most challenging data science problem you have encountered? What made it challenging?

**What they are testing:** depth, ownership, and judgment. Amazon lens: Ownership.

- **Situation and Task:** the problem and why it mattered, in two sentences.
- **Action:** name what made it hard (messy or missing labels, a leak found late, an unclear target definition, scale, stakeholders who disagreed) and what you decided and built, using "I."
- **Result:** the measurable outcome and what you would do differently.

**Pitfall:** describing the model instead of the difficulty.

### Tell me about a time you did not have enough data to make the right decision. What did you do, and was it the right call?

**What they are testing:** decisions under ambiguity. Amazon lens: Bias for Action, Are Right, A Lot.

- **Situation and Task:** the decision, its deadline, and what data was missing.
- **Action:** how you bounded the uncertainty (a proxy metric, a small experiment, expert input, a reversible first step), what you chose, and what you would check later.
- **Result:** the outcome, told honestly. If it was the wrong call, say what the data later showed and what you changed.

**Pitfall:** waiting for perfect data, or hiding that the call was uncertain.

### Tell me about a situation that required you to dig deep to find the root cause. How did you know you were focusing on the right things?

**What they are testing:** rigor and systematic thinking. Amazon lens: Dive Deep.

- **Situation and Task:** the symptom (a metric drop, a model degrading, a pipeline discrepancy).
- **Action:** the hypotheses you listed, how you tested and ruled each out (slicing the data, checking upstream tables, reproducing the issue), and how you knew you had the cause and not a correlation.
- **Result:** the fix, how you confirmed it, and what you would do differently or automate.

**Pitfall:** presenting the answer as obvious instead of showing the elimination.

### Give me an example of a complex problem you solved with a simple solution. How do you know it worked?

**What they are testing:** judgment and restraint. Amazon lens: Invent and Simplify.

- **Situation and Task:** what made the problem look complex, and what the complex options were.
- **Action:** why you chose the simple approach (a rule, a baseline, a logistic regression, a query change), and what you compared it against.
- **Result:** evidence it worked, such as a holdout comparison, an A/B test, or monitoring after launch, with numbers.

**Pitfall:** claiming simplicity without evidence that it matched the complex option.

**Before the interview:** write each story as a short bullet outline, pick numbers you can defend, and rehearse aloud. One good story can answer more than one of these prompts.
