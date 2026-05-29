# Data & Dataset

**Module:** `leaspy.io.data`

Leaspy separates the **user-facing data container** (`Data`) from the **computation-optimized representation** (`Dataset`).

## `Data`: The User Interface

The **`Data`** class is what you will use 99% of the time. It wraps your raw data (usually a Pandas DataFrame) and prepares it for usage with Leaspy.

*   **Flexible**: You can easily inspect, modify, slice, and reload variables (cofactors, headers).
*   **Convenient**: Methods like `Data.from_csv_file` or `Data.from_dataframe` handle the complex formatting logic for you.

## `Dataset`: The Computational Core

The **`Dataset`** class is an internal, read-only optimization of `Data`.

When you call `model.fit(data)`, Leaspy automatically converts your `Data` object into a `Dataset` behind the scenes. **You rarely need to instantiate this class yourself.**

### Why does it exist?
While `Data` is user-friendly, `Dataset` is machine-friendly:
1.  **Tensorized**: Converts everything to PyTorch tensors for fast math.
2.  **Locked**: Prevents accidental modification during training.
3.  **Optimized Layout**: Handles padding, masking for missing values, and memory layout for batch operations.

### When should you use `Dataset` directly?
The only real use case for manually creating a `Dataset` is for **performance** in advanced scripts. If you are running thousands of iterations or repeated fit/personalize calls on the **same static data**, you can convert it once:

```python
# Optimize once
dataset = Dataset(data)

# Reuse many times (avoids re-converting data at every call)
for i in range(100):
   model.fit(dataset, ...) 
```

## Where is Data Handled in the Code?

Users typically pass `Data` (or even a `pandas.DataFrame`) to the model's main methods: `fit`, `predict`, or `personalize`. 

The conversion happens in [`BaseModel`](../models/BaseModel.md), the ancestor of all Leaspy models (including `LogisticModel`):

1.  **`BaseModel.fit(data)`**: 
    *   Accepts `DataFrame`, `Data`, or `Dataset`.
    *   Calls internal helper `_get_dataset(data)`.
    *   `_get_dataset` converts `DataFrame` $\rightarrow$ `Data` $\rightarrow$ `Dataset` as needed.
2.  **Algorithm Execution**: The optimized `Dataset` is then passed to the algorithm runner (e.g., `algo.run(model, dataset)`).

This means high-level models like `LogisticModel` never worry about data formatting—they simply rely on `BaseModel` to hand them a clean, tensorized `Dataset`.
