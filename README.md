# Graph-Neural-Networks-for-Fraud-Detection



# **Fraud Detection in Payment Transactions using Graph Neural Networks**

## **Project Overview**
This project leverages **Graph Neural Networks (GNNs)** to detect fraudulent transactions in a bank's payment network. Traditional fraud detection models rely on tabular features, but fraud often occurs in complex networks where relationships matter. By modeling transactions as a **graph**, we can capture hidden fraud patterns that traditional methods miss.

We experimented with two state-of-the-art GNN architectures:
- **Graph Convolutional Network (GCN)**
- **Graph Attention Network (GAT)**

GAT outperformed GCN due to its ability to weigh node relationships dynamically.

---

## **Data & Graph Construction**
The dataset consists of financial transactions with attributes such as:
- `transaction_id`: Unique identifier for the transaction
- `sender`: Account initiating the transaction
- `receiver`: Account receiving the funds
- `amount`: Transaction amount
- `timestamp`: Time of the transaction
- `is_fraud`: Binary label (1 = Fraudulent, 0 = Non-Fraudulent)

### **Graph Representation**
We constructed a **directed graph** where:
- **Nodes represent bank accounts (senders/receivers).**
- **Edges represent transactions.**
- Edge attributes include `amount`, `timestamp`, and `is_fraud`.

---

## **Feature Engineering**
To enhance model performance, we engineered graph-based features:

| Feature | Description |
|---------|------------|
| `in_degree` | Number of incoming transactions to a node (account). |
| `out_degree` | Number of outgoing transactions from a node. |
| `pagerank` | Node importance based on transaction volume and connectivity. |
| `betweenness` | Measures how often a node lies on the shortest paths between other nodes (potential intermediaries in fraud). |
| `mean_amount` | Average transaction amount per node. |
| `std_amount` | Standard deviation of transaction amounts, indicating variability. |
| `fraud_score` | Number of fraudulent transactions associated with a node. |

These features help differentiate fraudulent behavior from normal transactions.

---

## **Model Development & Training**
We trained two GNN models:

### **Graph Convolutional Network (GCN)**
GCN aggregates node features from neighbors using a weighted sum approach. While effective, it treats all neighbors equally, limiting its ability to capture fraud relationships.

**Results:**
```
Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.92      0.95     56155
           1       0.16      0.64      0.26      1449

    accuracy                           0.91     57604
   macro avg       0.58      0.78      0.61     57604
weighted avg       0.97      0.91      0.93     57604
```
- Performed well on non-fraudulent transactions.
- Low precision and F1-score for fraudulent transactions indicate poor fraud detection.

### **Graph Attention Network (GAT)**
GAT improves upon GCN by assigning **attention weights** to neighbors, allowing the model to focus on **more relevant connections**.

**Results:**
```
Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.89      0.94     56155
           1       0.19      0.97      0.32      1449

    accuracy                           0.90     57604
   macro avg       0.59      0.93      0.63     57604
weighted avg       0.98      0.90      0.93     57604
```
- **Higher recall (0.97) for fraudulent transactions** compared to GCN.
- Successfully identified more fraudulent transactions but with slightly lower precision.
- More interpretable, as attention scores highlight risky connections.

---

## **Implementation**
### **Model Training**
```python
# Import necessary libraries
import torch
from torch_geometric.nn import GCNConv, GATConv

class FraudGCN(torch.nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(FraudGCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_features)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
```

```python
class FraudGAT(torch.nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, heads=4):
        super(FraudGAT, self).__init__()
        self.conv1 = GATConv(in_features, hidden_dim, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_dim * heads, out_features, heads=1, concat=False)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
```

### **Training Pipeline**
```python
from torch_geometric.loader import DataLoader

def train_model(model, data, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```
---

## **Conclusion**
- **GAT significantly improved fraud detection recall compared to GCN.**
- Future improvements: Try **GraphSAGE, GIN, or Heterogeneous Graph Models**.
- Further tuning can optimize precision-recall balance for fraud detection.

For inquiries, reach out at: vas7084@nyu.edu


