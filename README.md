# AI Conversion Project: Uplift Modeling

## 1. The Core Objective
We are building a machine learning system to maximize conversion rates using **Uplift Modeling** (specifically a **Two-Model Approach** or "T-Learner"). Instead of predicting *if* a user will convert, we are predicting the **incremental impact** of sending them a marketing message or showing an ad.

Our goal is to find the **"Persuadables"**—users whose probability of converting will significantly increase *only if* they receive the intervention.

## 2. The Problem We Are Solving
Traditional machine learning asks: *"Will this user buy?"* 
This leads to targeting people who were already going to buy anyway (wasting marketing budget) or annoying people who might leave because of the ad.

We are solving the problem of **optimizing marketing efficiency**. By shifting the question to *"Did our ad CAUSE them to buy?"*, we segment users into four buckets:
1. **The Persuadables:** Will *only* buy if we show them the ad. **(TARGET)**
2. **The Sure Things:** Will buy whether they see the ad or not. *(Do Not Target - Save money)*
3. **The Lost Causes:** Will never buy, regardless of the ad. *(Do Not Target - Save money)*
4. **The Sleeping Dogs:** Were going to buy, but an ad will annoy them into leaving. *(Avoid!)*

## 3. The Dataset
This project uses the **Criteo Uplift Modeling Dataset** (13 Million rows), generated through a massive Randomized Control Trial (A/B Test). 

* **`f0` - `f11` (Features):** The user's profile (e.g., past clicks, device type).
* **`treatment`:** 
  * `1` (Treated): Randomly selected to see the ad (84.6%).
  * `0` (Control): Randomly selected NOT to see the ad (15.4%).
* **`conversion` (Target):** Whether they actually bought something (`1`) or not (`0`).

Because the ad assignment was completely random, we can guarantee that the *only* difference between the two massive groups is whether or not they saw the ad. This allows our machine learning models to prove **causality**.

## 4. System Architecture (How It Works)

### Training Phase (The Two-Model Approach)
1. **`Model_T` (Treatment Model):** Trained only on users where `treatment = 1`. It learns what a converting user looks like when they **do** see an ad.
2. **`Model_C` (Control Model):** Trained only on users where `treatment = 0`. It learns what a converting user looks like when they **do not** see an ad.

### Inference Phase (The Decision Engine)
When a brand new user arrives, the system grabs their 12 features (`f0`-`f11`) and does the following in real-time:

1. **Ask `Model_T`:** "If we DO show this person an ad, what is the probability they will buy?" *(e.g., 8%)*
2. **Ask `Model_C`:** "If we DO NOT show this person an ad, what is their organic probability of buying?" *(e.g., 2%)*
3. **Calculate Uplift:** `8% - 2% = +6% Uplift`
4. **Make Decision:** Because the Uplift is greater than 0, the Decision Engine automatically triggers the action: **Send the Ad!**
