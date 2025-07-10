
# Attention Analysis Report for Sarcasm Detection

## Overview
- **Text**: "Congratulations on stating the obvious. I am sure glaciers will start moving any minute now"
- **Context**: ""
- **Prediction**: Sarcastic (Confidence: 0.9531)
- **Analyzed**: 3 layers, 12 heads per layer

## Key Findings

### Most Attentive Tokens
The following tokens receive the most attention:

| Token | Position | Self-Attention | Attention Focus |
|-------|----------|----------------|----------------|
| Congratulations | 0 | 0.1883 | 0.6189 |
| minute | 14 | 0.1591 | 0.6260 |
| . | 5 | 0.1520 | 0.5422 |
| moving | 12 | 0.1267 | 0.6467 |
| obvious | 4 | 0.1235 | 0.6203 |

### Attention Patterns

#### Strong Token Connections
The following token pairs show strong attention connections:

| From | To | Attention Weight |
|------|----|-----------------:|
| obvious | . | 0.2081 |
| on | Congratulations | 0.1801 |
| I | . | 0.1762 |
| stating | . | 0.1693 |
| will | . | 0.1651 |
