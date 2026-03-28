# Dataset Description — OECD Better Life Index (European Countries)

## 1. What is the domain of the problem about?

The problem concerns selecting the best European country to live in from the perspective of quality of life. The dataset captures multiple dimensions of well-being — economic performance, health, personal satisfaction, work-life balance, environmental quality, and geographic accessibility — to support a multi-criteria evaluation of European OECD member states as potential places to relocate.

## 2. What is the source of the data?

The primary data source is the **OECD Better Life Index** dataset, published by the Organisation for Economic Co-operation and Development (OECD). The dataset was obtained from Kaggle: https://www.kaggle.com/datasets/joebeachcapital/oecd-better-life-index. The underlying data originates from official OECD statistics, EU-SILC surveys, and national statistical offices. The data covers averages from the 2015–2020 period. The "Distance from Poznan" criterion was computed separately using the haversine formula based on capital city coordinates.

## 3. What is the point of view of the decision maker?

The decision maker is a **young person (student) based in Poznan, Poland**, evaluating European countries as potential destinations for relocation. The decision maker values economic opportunity (employment, earnings), health and well-being (life expectancy, life satisfaction), work-life balance (low overwork), environmental quality (air pollution), and geographic proximity to their home city of Poznan.

## 4. What is the number of alternatives considered? Were there more of them in the original data set?

The final dataset contains **26 alternatives** (European countries). The original OECD BLI dataset contains 42 entries (including non-European countries like Australia, Japan, USA, Brazil, etc., and an "OECD - Total" aggregate). After filtering to European OECD members, Turkiye was excluded due to missing "Personal earnings" data. The remaining 26 European countries all have complete data for all 8 selected criteria.

## 5. Describe one of the alternatives considered

**Norway**:
- Employment rate: 75.0% (gain — high, strong labor market)
- Long-term unemployment rate: 0.9% (cost — very low, excellent)
- Personal earnings: 55,780 USD/year (gain — among the highest)
- Life expectancy: 83.0 years (gain — very high)
- Life satisfaction: 7.3/10 (gain — high)
- Employees working very long hours: 1.4% (cost — very low, excellent work-life balance)
- Air pollution: 6.7 μg/m³ (cost — very low, excellent air quality)
- Distance from Poznan: 917.2 km (cost — moderate)

Norway is a strong all-round alternative with excellent scores on nearly every criterion. It dominates 5 other countries in the dataset (France, Greece, Ireland, Portugal, United Kingdom). Its only relative weakness is the moderate distance from Poznan and slightly lower life satisfaction compared to Finland (7.9).

## 6. What is the number of criteria considered? Were there more of them in the original data set?

The dataset uses **8 criteria**. The original OECD BLI dataset contains 24 indicators covering 11 well-being topics. Seven indicators were selected based on their relevance to the decision problem (quality of life for a young person considering relocation). The eighth criterion (Distance from Poznan) was added to capture geographic accessibility, which is not present in the original dataset.

## 7. What are the domains of the individual criteria? What is the nature (gain/cost)?

| # | Criterion | Domain | Range in dataset | Nature |
|---|-----------|--------|-----------------|--------|
| 1 | Employment rate | Continuous (%) | 56.0 – 80.0 | Gain |
| 2 | Long-term unemployment rate | Continuous (%) | 0.6 – 10.8 | Cost |
| 3 | Personal earnings | Continuous (USD/year) | 23,619 – 67,488 | Gain |
| 4 | Life expectancy | Continuous (years) | 75.5 – 84.0 | Gain |
| 5 | Life satisfaction | Continuous (scale 0–10) | 5.8 – 7.9 | Gain |
| 6 | Employees working very long hours | Continuous (%) | 0.3 – 11.7 | Cost |
| 7 | Air pollution | Continuous (μg/m³) | 5.5 – 22.8 | Cost |
| 8 | Distance from Poznan | Continuous (km) | 0.0 – 2,562.8 | Cost |

All criteria are expressed on continuous numerical scales.

## 8. Are all criteria of equal importance?

No, the criteria are not of equal importance. From the perspective of a young person considering relocation, the estimated weights (scale 1–10) are:

| Criterion | Weight (1–10) | Justification |
|-----------|:---:|---------------|
| Personal earnings | 9 | Financial stability is the primary concern for relocation |
| Employment rate | 8 | Job availability directly impacts quality of life |
| Life satisfaction | 8 | Overall well-being is a key motivator |
| Air pollution | 7 | Health and environment matter significantly |
| Life expectancy | 6 | Indicator of healthcare system quality |
| Long-term unemployment rate | 6 | Risk of prolonged joblessness is a concern |
| Distance from Poznan | 5 | Proximity to home matters but is not decisive |
| Employees working very long hours | 5 | Work-life balance matters but is secondary to economic factors |

No criterion is completely irrelevant — all were deliberately chosen for their relevance. "Employees working very long hours" and "Distance from Poznan" have the lowest weights but remain meaningful indicators of work-life balance and accessibility.

## 9. Are there dominated alternatives?

Yes, there are numerous dominated alternatives in the dataset. A selection of dominated pairs:

**Greece is dominated by 7 countries** (Belgium, Finland, Luxembourg, Netherlands, Norway, Sweden, Switzerland):

| Criterion | Greece | Netherlands | Norway | Sweden |
|-----------|--------|------------|--------|--------|
| Employment rate | 56.0 | 78.0 | 75.0 | 75.0 |
| Long-term unemp. | 10.8 | 0.9 | 0.9 | 1.0 |
| Personal earnings | 27,207 | 58,828 | 55,780 | 47,020 |
| Life expectancy | 81.7 | 82.2 | 83.0 | 83.2 |
| Life satisfaction | 5.8 | 7.5 | 7.3 | 7.3 |
| Long hours (%) | 4.5 | 0.3 | 1.4 | 0.9 |
| Air pollution | 14.5 | 12.2 | 6.7 | 5.8 |
| Distance (km) | 1,688.1 | 814.9 | 917.2 | 773.1 |

**Other notable dominated pairs:**
- Austria dominates Slovenia
- Denmark dominates Hungary, Latvia, and Slovak Republic
- Germany dominates Slovak Republic
- Norway dominates France, Greece, Ireland, Portugal, and United Kingdom
- Sweden dominates Estonia, France, Greece, and Portugal
- Switzerland dominates France, Greece, and Italy
- Luxembourg dominates Belgium and Greece
- Netherlands dominates Belgium and Greece
- Finland dominates Greece and Portugal

In total, **24 dominance relationships** were identified. Greece is the most dominated alternative (dominated by 7 countries), while Norway and Sweden are the strongest dominators (each dominating 5 and 4 countries respectively).

## 10. What should the theoretically best alternative look like?

The theoretically best alternative would combine a **strong advantage on key economic criteria** (high personal earnings, high employment rate, low unemployment) with **good health outcomes** (high life expectancy) and **high life satisfaction**. It should also have low air pollution, low overwork, and be relatively close to Poznan.

Specifically, the ideal profile would be:
- Employment rate: ~80% (like Switzerland)
- Long-term unemployment rate: ~0.6% (like Poland/Czech Republic)
- Personal earnings: ~67,488 USD (like Iceland)
- Life expectancy: ~84.0 years (like Switzerland)
- Life satisfaction: ~7.9 (like Finland)
- Employees working very long hours: ~0.3% (like Netherlands)
- Air pollution: ~5.5 μg/m³ (like Finland)
- Distance from Poznan: ~0 km (like Poland) or ~238.8 km (like Germany)

A strong advantage on earnings and employment combined with good environmental quality would be more valuable than marginal improvements across all criteria, because economic factors have the highest decision weight.

## 11. Which alternative seems to be the best?

**Switzerland** appears to be the best overall alternative:

| Criterion | Switzerland | Rank (out of 26) |
|-----------|------------|:-:|
| Employment rate | 80.0% | 1st |
| Long-term unemp. | 1.7% | 12th |
| Personal earnings | 64,824 USD | 2nd |
| Life expectancy | 84.0 years | 1st |
| Life satisfaction | 7.5 | 3rd (tied) |
| Long hours (%) | 0.4% | 2nd |
| Air pollution | 10.1 μg/m³ | 11th |
| Distance (km) | 911.6 | 12th |

Switzerland leads on the two most impactful criteria — highest employment rate (80%) and highest life expectancy (84.0 years) — while also ranking 2nd in personal earnings (64,824 USD) and 2nd in work-life balance (only 0.4% working very long hours). Its strength is determined by the **overall excellence across criteria** rather than a single dominant factor. Its weaknesses are average air pollution (10.1 μg/m³, 11th) and moderate distance from Poznan (911.6 km). Switzerland dominates 3 countries (France, Greece, Italy).

Close competitors are **Netherlands** (1st in earnings among non-Iceland countries, lowest overwork at 0.3%) and **Norway** (excellent all-round with the best air quality among top performers at 6.7 μg/m³).

## 12. Which alternative seems to be the worst?

**Greece** is clearly the worst alternative:

| Criterion | Greece | Rank (out of 26) |
|-----------|--------|:-:|
| Employment rate | 56.0% | 26th (last) |
| Long-term unemp. | 10.8% | 26th (last) |
| Personal earnings | 27,207 USD | 23rd |
| Life expectancy | 81.7 years | 14th |
| Life satisfaction | 5.8 | 26th (last) |
| Long hours (%) | 4.5% | 15th |
| Air pollution | 14.5 μg/m³ | 19th |
| Distance (km) | 1,688.1 | 22nd |

Greece ranks last on three criteria simultaneously (employment rate, long-term unemployment, life satisfaction) and is dominated by 7 other countries — the most dominated alternative in the entire dataset. Its poor standing is determined by the **overall weakness across criteria** — especially the devastating combination of lowest employment (56%), highest long-term unemployment (10.8%), and lowest life satisfaction (5.8). Its only relative strength is a decent life expectancy (81.7 years, 14th), which is close to the European average.

## 13. Pairwise comparisons between alternatives

**Comparison 1: Switzerland > Netherlands**
- Switzerland better on: Employment rate (80 vs 78), Personal earnings (64,824 vs 58,828), Life expectancy (84.0 vs 82.2), Air pollution (10.1 vs 12.2)
- Netherlands better on: Long-term unemp. (0.9 vs 1.7), Long hours (0.3 vs 0.4), Distance (814.9 vs 911.6)
- Tied on: Life satisfaction (7.5 vs 7.5)

**Comparison 2: Germany > Czech Republic**
- Germany better on: Employment rate (77 vs 74), Personal earnings (53,745 vs 29,885), Life expectancy (81.4 vs 79.3), Life satisfaction (7.3 vs 6.9), Long hours (3.9 vs 4.5), Air pollution (12.0 vs 17.0), Distance (238.8 vs 311.7)
- Czech Republic better on: Long-term unemp. (0.6 vs 1.2)

**Comparison 3: Denmark > Sweden**
- Denmark better on: Long-term unemp. (0.9 vs 1.0), Personal earnings (58,430 vs 47,020), Life satisfaction (7.5 vs 7.3), Distance (461.5 vs 773.1)
- Sweden better on: Employment rate (75 vs 74), Life expectancy (83.2 vs 81.5), Long hours (0.9 vs 1.1), Air pollution (5.8 vs 10.0)

**Comparison 4: Norway > United Kingdom** (dominance)
- Norway better on: Personal earnings (55,780 vs 47,147), Life expectancy (83.0 vs 81.3), Life satisfaction (7.3 vs 6.8), Long hours (1.4 vs 10.8), Air pollution (6.7 vs 10.1), Distance (917.2 vs 1,170.1)
- Tied on: Employment rate (75 vs 75), Long-term unemp. (0.9 vs 0.9)

**Comparison 5: Finland > Poland**
- Finland better on: Employment rate (72 vs 69), Personal earnings (46,230 vs 32,527), Life expectancy (82.1 vs 78.0), Life satisfaction (7.9 vs 6.1), Long hours (3.6 vs 4.2), Air pollution (5.5 vs 22.8)
- Poland better on: Long-term unemp. (0.6 vs 1.2), Distance (0.0 vs 993.3)
