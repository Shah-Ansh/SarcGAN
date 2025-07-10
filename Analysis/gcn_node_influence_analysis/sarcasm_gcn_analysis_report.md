
# GCN Node Influence Analysis for Sarcasm Detection

## Overview
- **Text**: "Congratulations on stating the obvious. I am sure glaciers will start moving any minute now"
- **Context**: "None"
- **Prediction**: Sarcastic (Confidence: 0.9583)
- **Number of Nodes**: 16 (after filtering)
- **Tokens Analyzed**: 16

## Key Findings

### Most Important Nodes
The following tokens have the highest importance in the final GCN layer:

| Node | Token | Importance | Growth Rate |
|------|-------|------------|-------------|
| 5 | congratulations | 10.3979 | 2.13x |
| 6 | on | 7.9881 | 1.61x |
| 13 | sure | 7.7580 | 1.46x |
| 17 | moving | 7.6349 | 1.26x |
| 7 | stating | 7.6333 | 1.54x |

### Node Importance Evolution
The following shows how node importance evolves across GCN layers:

| Node | Token | Layer 1 | Layer 2 | Layer 3 | Layer 4 | Growth Pattern |
|------|-------|---------|---------|---------|---------|----------------|
| 5 | congratulations | 4.8801 | 10.8211 | 12.0764 | 10.3979 | Increasing |
| 6 | on | 4.9548 | 8.5397 | 9.9492 | 7.9881 | Increasing |
| 13 | sure | 5.3124 | 8.4594 | 11.0017 | 7.7580 | Increasing |
| 17 | moving | 6.0536 | 7.7283 | 9.5366 | 7.6349 | Increasing |
| 7 | stating | 4.9540 | 7.4506 | 9.4690 | 7.6333 | Increasing |
